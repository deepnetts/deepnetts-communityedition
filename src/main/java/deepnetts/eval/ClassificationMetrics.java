package deepnetts.eval;

import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import javax.visrec.ml.eval.EvaluationMetrics;

/**
 * Various metrics that tell us how good is a classifier.
 * Calculates various classification metrics which are used for classifier evaluation.
 * For multi class classification enables setting to which specific class values refer to.
 * 
 * @see ConfusionMatrix
 * @see Evaluators
 */
public final class ClassificationMetrics extends EvaluationMetrics {

//  * https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
    // todo: add Cohen's Kappa and others from the above link, 
    // also balanced accuracy
    
    private String classLabel;    
    
    private final float truePositive;
    private final float trueNegative;
    private final float falsePositive;
    private final float falseNegative;
    private final float total;
    
    private ConfusionMatrix confusionMatrix;
    private int classIdx;
    
    private static final HashMap<String, String> descriptions;
    
    static {
        descriptions = new HashMap<>();
        descriptions.put(ACCURACY, "How often is a classifier correct in total (percent of correct classifications)");
        descriptions.put(PRECISION, "How often is a classifier correct when it gives positive prediction");
        descriptions.put(RECALL, "When it is actually positive class, how often does it give positive prediction");
        descriptions.put("Specificity", "When it is actually negative class, how often does it give negative prediction");
        descriptions.put("FalsePositiveRate", "How often it gives false positive prediction in total (percent of false positive predictions)");
        descriptions.put("FalseNegativeRate", "How often it gives false negative prediction in total (percent of false negative predictions)");
        descriptions.put(F1SCORE, "Harmonic average (balance) of precision and recall");
        descriptions.put(ConfusionMatrix.TRUE_POSITIVE , "Number of examples correctly classified as positive");
        descriptions.put(ConfusionMatrix.TRUE_NEGATIVE, "Number of examples correctly classified as negative");
        descriptions.put(ConfusionMatrix.FALSE_POSITIVE, "Number of examples incorrectly classified as positive");
        descriptions.put(ConfusionMatrix.FALSE_NEGATIVE, "Number of examples incorrectly classified as negative");
    }    
         
    /**
    * Constructs a new classification metrics from specified confusion matrix.
    * 
    * @param confMatrix confusion matrix to extract metrics from.
    */    
    public ClassificationMetrics(ConfusionMatrix confMatrix) {
        this.confusionMatrix = confMatrix;
        
        // kada se instancira za multi class treba da sabira dijagonalu
        this.truePositive = confMatrix.getTruePositive();
        this.trueNegative = confMatrix.getTrueNegative();
        this.falsePositive = confMatrix.getFalsePositive();
        this.falseNegative = confMatrix.getFalseNegative();
        this.total = falseNegative + falsePositive + trueNegative + truePositive;
        
        this.set(ConfusionMatrix.TRUE_POSITIVE, truePositive);
        this.set(ConfusionMatrix.TRUE_NEGATIVE, trueNegative);
        this.set(ConfusionMatrix.FALSE_POSITIVE, falsePositive);
        this.set(ConfusionMatrix.FALSE_NEGATIVE, falseNegative);  
        
        this.set("TotalCorrect", truePositive + trueNegative);
        this.set("TotalIncorrect", falsePositive + falseNegative);        
        
    }
    
    /**
     * Constructs a new classification metrics of a single class for multi class classification.
     * 
     * @param confMatrix
     * @param classLabel
     * @param classIdx 
     */
    public ClassificationMetrics(ConfusionMatrix confMatrix, String classLabel, int classIdx) {
        this.classLabel = classLabel;
        this.classIdx =classIdx;
        this.confusionMatrix = confMatrix;
        this.truePositive = confMatrix.getTruePositive(classIdx);
        this.trueNegative = confMatrix.getTrueNegative(classIdx);
        this.falsePositive = confMatrix.getFalsePositive(classIdx);
        this.falseNegative = confMatrix.getFalseNegative(classIdx);
        this.total = falseNegative + falsePositive + trueNegative + truePositive; // + none? ovo toal nije tacno treba sabrati sve! Celu matricu da li da imam none uopste? gde njih da racunam

        this.set(ConfusionMatrix.TRUE_POSITIVE, truePositive);
        this.set(ConfusionMatrix.TRUE_NEGATIVE, trueNegative);
        this.set(ConfusionMatrix.FALSE_POSITIVE, falsePositive);
        this.set(ConfusionMatrix.FALSE_NEGATIVE, falseNegative);   
    }    

   /**
    * Constructs a new classification metrics using specified arguments.
    *
    * @param truePositive
    * @param trueNegative
    * @param falsePositive
    * @param falseNegative
    */
    public ClassificationMetrics(int trueNegative, int falsePositive, int falseNegative, int truePositive) {
        this.truePositive = truePositive;
        this.trueNegative = trueNegative;
        this.falsePositive = falsePositive;
        this.falseNegative = falseNegative;
        this.total = falseNegative + falsePositive + trueNegative + truePositive;
        
        this.set(ConfusionMatrix.TRUE_POSITIVE, truePositive);
        this.set(ConfusionMatrix.TRUE_NEGATIVE, trueNegative);
        this.set(ConfusionMatrix.FALSE_POSITIVE, falsePositive);
        this.set(ConfusionMatrix.FALSE_NEGATIVE, falseNegative);           
    }    
  
    
    /**
     * Returns class label that these metric correspond to (used for multi class classification).
     * In case you have multiple classes each class has its classification metrics.
     * 
     * @return class label
     */
    public String getClassLabel() {
        return classLabel;
    }
       

    /**
     * Sets class label to which this metrics corresponds too
     * @param classLabel 
     */
    public void setClassLabel(String classLabel) {
        this.classLabel = classLabel;
    }

    public int getClassIdx() {
        return classIdx;
    }
    
    
    
    /**
     * Returns a confusion matrix that is used to generate these metrics.
     * 
     * @return 
     */
    public ConfusionMatrix getConfusionMatrix() {
        return confusionMatrix;
    }    

    /**
     * Percent of correct classifications (for both positive and negative classes).
     * Answers the question how often a classifier gives correct answer.
     * Accuracy is a good measure classes in the data are nearly balanced.
     * This metric might be misleading if the classes are not balanced.
     * 
     * Accuracy = ( TruePositive + TrueNegative ) / Total
     *
     * @return how often is the classifier correct
     */
    public float getAccuracy() {
        if (total == 0) return 0;
        return (truePositive + trueNegative) / total;
    }

    /**
     * A percent of wrong classifications/predictions made.
     * Answers the question how often a classifier gives wrong answer?
     * 
     * error = (fp + fn) / total
     * error = 1 - accuracy
     *
     * @return classification error rate
     */
    public float getErrorRate() {
        if (total == 0) return 0;        
        return (falsePositive + falseNegative) / total;
    }


    /**
     * What percent of those predicted as positive are really positive.
     * Answers the question: when it predicts yes, how often is it correct?
     *
     * precision = truePositive / (truePositive + falsePositive)
     *
     * @return percent of those predicted as positive that are really positive.
     */
    public float getPrecision() {
        if (total == 0) return 0;        
        return truePositive / (truePositive + falsePositive);
    }

    /**
     * Ratio between those classified as positive compared to those that are actually positive.
     * Also called Sensitivity or True Positive Rate. 
     * 
     * @return how often classifier predicts yes, when actual class is yes
     */
    public float getRecall() {
        if (total == 0) return 0;        
         return truePositive / (truePositive + falseNegative);
    }

    /**
     * Specificity or true negative rate.
     * When it's actually no, how often does it predict no?
     * 
     * @return
     */
    public float getSpecificity() {
        return trueNegative / (trueNegative + falsePositive);
    }

   /**
    * Calculates and returns F1 score - a balance between recall and precision.
    * f1 = 2  * ( (precision*recall) / (precision+recall))
    * 
    * @return f-score metric (harmonic average of recall and precision)
    */
    public float getF1Score() {
        float precision = getPrecision();
        float recall = getRecall();
        
        final float f1 = 2 * ((precision * recall) / (precision + recall));
        
        if (Double.isNaN(f1))
            return 0;
        else
            return f1;        
    }
    
    /**
     * Balance between precision and recall.
     * 
     * @param beta
     * @return f-score
     */
    public float getFScore(int beta) {
        float precision = getPrecision();
        float recall = getRecall();        
        
        float f = ((beta * beta + 1) * precision * recall)
                / (float)(beta * beta * precision + recall);
        
        if (Double.isNaN(f))
            return 0;
        else
            return f;
    }    

    /**
     * Returns total number of classifications.
     *
     * @return total number of classifications
     */
    public int getTotal() {
        return (int)total;
    }

    /**
     * When it's actually no, how often does it predict yes?
     * FP/actual no
     * @return
     */
    public float getFalsePositiveRate() {
        return falsePositive / (falsePositive + trueNegative);
    }
        
    /**
     * How often does positive class actually occur in the sample
     * @return 
     */
    public float positiveFreqency() {
        return (truePositive+falseNegative) / total;
    }

    /**
     * How often does negative class actually occur in the sample
     * @return 
     */
    public float negativeFreqency() {
        return (trueNegative+falsePositive) / total;
    }    

    /**
     * When its actually yes, how often does it predicts no
     * @return
     */
    public float getFalseNegativeRate() {
        return falseNegative / (falseNegative + truePositive );
    }

    /**
     * When its actually no, how often it is classified as yes
     * @return
     */
    public float getFalseDiscoveryRate() {
        return falsePositive / (falsePositive+truePositive);
    }
        
    /**
     * Calculates and returns the matthews corellation coefficient.
     * The F1 metric is not a suitable method of combining precision and recall i
     * measure of the quality of binary (two-class) classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
     * The coefficient takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.[5] The MCC is in essence a correlation coefficient between the observed and predicted binary classifications; it returns a value between −1 and +1. A coefficient of +1 represents a perfect prediction, 0 no better than random prediction and −1 indicates total disagreement between prediction and observation.
     *  
     * @return value of matthews correlation coeffiicent
     * 
     * @see http://en.wikipedia.org/wiki/Matthews_correlation_coefficient    
     */
    public double getMatthewsCorrelationCoefficient() {
        return (truePositive * trueNegative - falsePositive * falseNegative) /
                (sqrt((truePositive + falsePositive) * (truePositive + falseNegative) * (trueNegative + falsePositive) * (trueNegative + falseNegative)));
    }
    
    /**
     * Balanced accuracy is a good metric to use when data set is not balanced.
     * It is an average of specificity and recall(sensitivity)
     * @return 
     */  
    public double getBalancedAccuracy() {
        if (trueNegative == 0 && falsePositive == 0)
            return truePositive / (truePositive + falseNegative);
        
        if (truePositive == 0 && falseNegative == 0)
            return trueNegative / (trueNegative + falsePositive);

        return 0.5 * (truePositive / (truePositive + falseNegative) + trueNegative / (trueNegative + falsePositive));
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("Class: ").append(classLabel).append(System.lineSeparator());
        sb.append("Total items: ").append(getTotal()).append(System.lineSeparator());
        sb.append("True positive:").append(truePositive).append(" ").append(descriptions.get(ConfusionMatrix.TRUE_POSITIVE)).append(System.lineSeparator());
        sb.append("True negative:").append(trueNegative).append(" ").append(descriptions.get(ConfusionMatrix.TRUE_NEGATIVE)).append(System.lineSeparator());
        sb.append("False positive:").append(falsePositive).append(" ").append(descriptions.get(ConfusionMatrix.FALSE_POSITIVE)).append(System.lineSeparator());
        sb.append("False negative:").append(falseNegative).append(" ").append(descriptions.get(ConfusionMatrix.FALSE_NEGATIVE)).append(System.lineSeparator());
        sb.append("Accuracy (ACC): ").append(getAccuracy()).append(" ").append(descriptions.get(ACCURACY)).append(System.lineSeparator());
        sb.append("Precision (PPV): ").append(getPrecision()).append(" ").append(descriptions.get(PRECISION)).append(System.lineSeparator());
        sb.append("Recall: ").append(getRecall()).append(" ").append(descriptions.get(RECALL)).append(System.lineSeparator());
        sb.append("F1 Score: ").append(getF1Score()).append(" ").append(descriptions.get(F1SCORE)).append(System.lineSeparator());        
        sb.append("Specificity (TNR): ").append(getSpecificity()).append(" ").append(descriptions.get("Specificity")).append(System.lineSeparator());
        sb.append("Fall-out (FPR): ").append(getFalsePositiveRate()).append(" ").append(descriptions.get("FalsePositiveRate")).append(System.lineSeparator());
        sb.append("False negative rate (FNR): ").append(getFalseNegativeRate()).append(" ").append(descriptions.get("FalseNegativeRate")).append(System.lineSeparator());
        // removed because tn is zero and then it gets negative
        //   sb.append("Matthews correlation Coefficient (MCC): ").append(getMatthewsCorrelationCoefficient()).append(" ").append("1=perfect, 0=same as random guess, -1=very bad ").append(System.lineSeparator());
        return sb.toString();
    }


    /**
     * Average values of commonly used classification metrics.
     */
    public final static class Stats {
        public double accuracy=0;
        public double precision=0;
        public double recall=0;
        public double fScore=0;
        public double mserror=0; // classification?
        public double correlationCoefficient = 0; // stats?

        @Override
        public String toString() {
            return "Stats{" + "accuracy=" + accuracy + ", precision=" + precision + ", recall=" + recall + ", fScore=" + fScore + ", mserror=" + mserror + ", corelationCoefficient=" + correlationCoefficient + '}';
        }
    }

    /**
     * Creates classification metrics from the given confusion matrix.
     * Creates an array of ClassificationMetrics objects one for each class.
     *
     * @param confusionMatrix
     * @return classification metrics
     */
    public static ClassificationMetrics[] createFrom(ConfusionMatrix confusionMatrix) {

        int classCount = confusionMatrix.getClassCount();
        if (classCount == 2) { // binary classification
            ClassificationMetrics[] measures = new ClassificationMetrics[1];
            String[] classLabels = confusionMatrix.getClassLabels();

                int tp = confusionMatrix.getTruePositive();
                int tn = confusionMatrix.getTrueNegative();
                int fp = confusionMatrix.getFalsePositive();
                int fn = confusionMatrix.getFalseNegative();

            measures[0] = new ClassificationMetrics(tn, fp, fn, tp);
            measures[0].setClassLabel(classLabels[0]);

            return measures;

        } else { // multiclass classification
            ClassificationMetrics[] measures = new ClassificationMetrics[classCount];
            String[] classLabels = confusionMatrix.getClassLabels();

            for(int clsIdx=0; clsIdx<confusionMatrix.getClassCount(); clsIdx++) { // for each class
                // ove metode mozda ubaciti u matricu Confusion matrix - najbolje tako
                int tp = confusionMatrix.getTruePositive(clsIdx);
                int tn = confusionMatrix.getTrueNegative(clsIdx);
                int fp = confusionMatrix.getFalsePositive(clsIdx);
                int fn = confusionMatrix.getFalseNegative(clsIdx);

                measures[clsIdx] = new ClassificationMetrics(tn, fp, fn, tp);
                measures[clsIdx].setClassLabel(classLabels[clsIdx]);
            }
            return measures;
        }

    }



    /**
     *
     * @param results list of different metric results computed on different sets of data
     * @return average metrics computed different MetricResults
     */
    public static ClassificationMetrics.Stats average(ClassificationMetrics[] results) {
        List<String> classLabels = new ArrayList<>();
         ClassificationMetrics.Stats average = new ClassificationMetrics.Stats();
          double count = 0;
            for (ClassificationMetrics cm : results) {
                average.accuracy += cm.getAccuracy();
                average.precision += cm.getPrecision();
                average.recall += cm.getRecall();
                average.fScore += cm.getF1Score();
//                average.mserror += er.getMeanSquareError();

                if(!classLabels.contains(cm.getClassLabel()))
                    classLabels.add(cm.getClassLabel());
            }
            count++;

        count = count * classLabels.size(); // * classes count
        average.accuracy = average.accuracy / count;
        average.precision = average.precision / count;
        average.recall = average.recall / count;
        average.fScore = average.fScore / count;
        average.mserror = average.mserror / count;

        return average;
    }


//    private static double[] createFScoresForEachClass(double[] precisions, double[] recalls) {
//        double[] fScores = new double[precisions.length];
//
//        for (int i = 0; i < precisions.length; i++) {
//            fScores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]);
//        }
//
//        return fScores;
//    }


//    private static double safelyDivide(double x, double y) {
//        double divisor = x == 0.0 ? 1 : x;
//        double divider = y == 0.0 ? 1.0 : y;
//        return divisor / divider;
//    }


}