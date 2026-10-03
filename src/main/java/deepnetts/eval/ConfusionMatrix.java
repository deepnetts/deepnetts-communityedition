package deepnetts.eval;

/**
 * Confusion matrix contains raw classifier test results.
 * It counts number of true and false predictions with respect to actual/target class of the given examples in test(evaluation) set.
 * Rows correspond to actual/target classes, and columns to predicted
 * 
 *                    Predicted
 *                      F   T
 *  Actual/target  F   TN  FP
 *  Actual/target  T   FN  TP
 *
 * https://en.wikipedia.org/wiki/Confusion_matrix
 */
public class ConfusionMatrix {
/*
 * TODO: linkovi na deep netts za ojasnjanje terninlogije
 * http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
 * http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix   
 * https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
 */
    
    /**
     * Class labels.
     */
    private final String[] classLabels;

    /**
     * Values in confusion matrix.
     */
    private final int[][] values;

    /**
     * Number of classes.
     */
    private final int classCount;

    /**
     * Total number of items classified in this matrix.
     * Sum of all matrix values
     */
    private int totalItems = 0;

    /**
     * Default setting for formating toString
     */
    private static final int STRING_DEFAULT_WIDTH = 7;

    /**
     * Creates a new confusion matrix for specified class labels
     * @param classLabels
     */
    public ConfusionMatrix(String[] classLabels) {

        if (classLabels == null) throw new IllegalArgumentException("Class labels cannot be null!");

        if (classLabels.length < 2) throw new IllegalArgumentException("Class labels cannot be less then 2!");

        for(String label : classLabels)
            if ((label == null) || label.isEmpty()) throw new IllegalArgumentException("Class label cannot be null or empty String!");

        this.classLabels = classLabels;
        classCount = classLabels.length;
        this.values = new int[classCount][classCount]; // what about negative? must be included in class labels
    }

    /**
     * Returns a value of confusion matrix at specified position.
     * @param actualIdx target/actual class idx  - corresponds to column
     * @param predictedIdx predicted class idx - corresponds to row
     * @return value of confusion matrix at specified position
     */
    public final int get(final int actualIdx, final int predictedIdx) {
       return values[actualIdx][predictedIdx];
    }

    /**
     * Increments matrix value at specified position.
     *
     * @param actualIdx class idx of actual class - corresponds to row
     * @param predictedIdx class idx of predicted class - corresponds to column
     *
     */
    public final void inc(final int actualIdx, final int predictedIdx) {
        values[actualIdx][predictedIdx]++;
        totalItems++; 
    }

    public final int getClassCount() {
        return classCount;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        int maxColumnWidth = STRING_DEFAULT_WIDTH;
        for (String label : classLabels)
            maxColumnWidth = Math.max(maxColumnWidth, label.length());

        maxColumnWidth+=2;
        
        builder.append("Confusion Matrix [actual, predicted]\n");
        // append column names
        builder.append(String.format("%1$" + maxColumnWidth + "s", "")).append(" |");
        for (String label : classLabels)
            builder.append(String.format("%1$" + maxColumnWidth + "s", label)).append(" |");
        builder.append("\n");
        // classLabels.length * maxColumnWidth  // none i actual
        
        // line
        for(int l=0; l< (classLabels.length+1) * (maxColumnWidth+2); l++)
            builder.append("-");

        builder.append("\n");
        
        for (int i = 0; i < values.length; i++) {
            builder.append(String.format("%1$" + maxColumnWidth + "s", classLabels[i])).append(" |");
            for (int j = 0; j < values[0].length; j++) {
                builder.append(String.format("%1$" + maxColumnWidth + "s", values[i][j])).append(" |");
            }
            builder.append("\n");

        }
        return builder.toString();
    }

    /**
     * Return true positive metric for binary classification.
     * True positives metric tells us percent of positive examples which are recognized by the classifier as positive.
     * Or in other words percent of correct predictions for the given positive examples. 
     * @return true positive metric for binary classification
     */
    public int getTruePositive() {
        return values[1][1];
    }


    /**
     * Returns true positive metric for specified class idx for multiclass classification.
     * True positive metric tells how many examples are correctly classified as a positive examples of the given class.
     * @param clsIdx Index of class for which true positive value is returned
     * @return
     */
    public int getTruePositive(int clsIdx) {
        return values[clsIdx][clsIdx];
    }

    public int getTrueNegative() {
        return values[0][0];
    }
    /*
    https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
*/
    // saberi sva ostala polja, a izuzmi red i kolonu  za zadatu klasu
    // trebalo bi zapravo sabrati sva druga po dijagonali? kako racunati true neagtive za mulsti class classification a da ne dize tacnostt vestacki?
    // all non-ci instances that are not classified as c1 - sum everything just skip ci row and col
    public int getTrueNegative(int clsIdx) {
        int trueNegative = 0;

//        for(int i = 0; i < classCount; i++) {
//            if (i == clsIdx) continue;
//            for(int j = 0; j < classCount; j++) {
//                if (j == clsIdx) continue;
//                trueNegative += values[i][j];
//            }
//        }
        trueNegative = values[0][0]; // negativni primeri
        return trueNegative;
    }

    /**
     * Returns number of false positive classifications.
     * Items that do not belong to specific class, but they are recognized as they do 
     * Only for binary classification
     * @return 
     */
    public int getFalsePositive() {
        return values[0][1];
    }

    // saberi celu clsIdx kolonu samo preskoci clsIdx red
    // all non-clsIdx that are classified/predicted as clsIdx
    public int getFalsePositive(int clsIdx) {
        int falsePositive = 0;
        // ovo mi nije dobro za 1 npr!s?
        for(int i=0; i<classCount; i++) { // ukljuci i nultu/none klasu - one koji su predicted kao clasIdx a zapravo su none
            if (i == clsIdx) continue; // skip tp value at diagonal
            falsePositive += values[i][clsIdx];
        }

        return falsePositive;
    }

    // Doublechecked: 13.4.19.
    // saberi ceo red  u kome se nalazi zadati clsIdx. znaci clsIdx red a preskoci kolonu clsIdx
    // all clsIdx that are classified as non clasIdx
    public int getFalseNegative(int clsIdx) {
        int falseNegative = 0;

        for(int i=0; i<classCount; i++) {
            if (i == clsIdx) continue; // skip tp value at diagonal
            falseNegative += values[clsIdx][i]; // sum all cells in the clsIdx row
        }
        
        // oni koji su klasifikovani kao none a actualy su druge klase su takodje false negative! 
        // on ih zapravo i dodaje jer krece od nule

        return falseNegative;
    }

    /**
     * How many positive items has been (falsely) classified as negative.
     * @return How many positive items has been (falsely) classified as negative
     */
    public int getFalseNegative() {
        return values[1][0];
    }

    public final String[] getClassLabels() {
        return classLabels;
    }

    public int getTotalItems() {
        return totalItems;
    }

    /**
     * A label for items classified as positive which are really positive.
     */
    public static final String TRUE_POSITIVE="TruePositive";
    
    /**
     * A label for items classified as negative which are really negative.
     */
    public static final String TRUE_NEGATIVE="TrueNegative";
    
    /**
     * A label for items falsely classified as positive, which are actually negative.
     */
    public static final String FALSE_POSITIVE="FalsePositive";
    
    
    /**
     * A label for items falsely classified as negative, which are actually positive.
     */
    public static final String FALSE_NEGATIVE="FalseNegative";

}