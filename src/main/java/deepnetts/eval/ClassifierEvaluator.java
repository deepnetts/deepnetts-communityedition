/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */
package deepnetts.eval;

import deepnetts.net.NeuralNetwork;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import javax.visrec.ml.eval.Evaluator;
import deepnetts.data.MLDataItem;
import deepnetts.tensor.Tensor;
import deepnetts.tensor.Tensor2D;
import deepnetts.util.Debugger;
import java.util.Collection;

/**
 * Evaluation method for binary and multi-class classifiers.
 * Calculates classification performance metrics, how good is it at predicting class of something.
 *
 * http://www.ritchieng.com/machine-learning-evaluate-classification-model/
 * http://scikit-learn.org/stable/modules/model_evaluation.html
 * http://notesbyanerd.com/2014/12/17/multi-class-performance-measures/
 * https://en.wikipedia.org/wiki/Confusion_matrix
 * https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
 * http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
 *
 * Micro and macro averaging: https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
 * I'm doing macro
 *

 */                                         
public class ClassifierEvaluator implements Evaluator<NeuralNetwork, DataSet<? extends MLDataItem>> { // use Classifier as a generic, wrap convolutional network with classifier

    /**
     * Constants used as labels for binary classification
     */
    private final static String LABEL_POSITIVE = "positive";
    private final static String LABEL_NEGATIVE = "negative";
    private final static String LABEL_NONE     = "none";

    /**
     * Class labels
     */
    private final List<String> classLabels = new ArrayList<>();

    /**
     * Confusion matrix that holds classification results
     */
    private ConfusionMatrix confusionMatrix;
    

    /**
     * Performance measures for each class, used for multi class classification
     */
    private HashMap<String, ClassificationMetrics> metricsByClass;

    /**
     * Classification threshold
     */
    private float threshold = 0.5f;

    private void init() {
        metricsByClass = new HashMap<>();

        if (classLabels.size() == 2) { // for binary classification - these should change positins?
            confusionMatrix = new ConfusionMatrix(new String[]{LABEL_NEGATIVE, LABEL_POSITIVE}); // labels for binary classification
        } else { // for multi class classification
            confusionMatrix = new ConfusionMatrix(classLabels.toArray(new String[classLabels.size()]));
            classLabels.forEach((label) -> {
                metricsByClass.put(label, null);
            });
        }
    }

    /**
     * Performs classifier evaluation and returns classification performance metrics.
     * @param neuralNet
     * @param testSet
     * @return 
     */
    @Override
    public  ClassificationMetrics evaluate(NeuralNetwork neuralNet, DataSet<? extends MLDataItem> testSet) { // NeuralNetwork, DataSet<?>
        classLabels.clear();
        classLabels.add(0, LABEL_NONE); // da li da dodajem negativnu klasu, vidi kako radi sci kit learn
        for(String label : testSet.getTargetColumnsNames()) { // ali nek uradi ovo samo jednom a ne za svaku epohu
            classLabels.add(label);
        }
        
        // if class labels are empty create class1, class2, classk ....
        init(); // ovo nekako raditi samo jednom kad se iterira

        //  wrap neural network with classifier interface, setInput from param, and return output
        // I need a method that wraps modelinto a classifier Classifier.fromNeuralNetwork(neuralNet)
        // so this method can accept classifier as aparam
        for (MLDataItem item : testSet) {
            final Tensor predictedOut = neuralNet.predict(item.getInput());
            
            if (predictedOut instanceof Tensor2D) {
                processResult((Tensor2D)item.getTargetOutput(), (Tensor2D)predictedOut);
            } else {
                processResult(item.getTargetOutput().getValues(), predictedOut.getValues());
            }
        }

        if (classLabels.size() == 2) {  // for binary classification
            return createBinaryClassifierMetrics();
        } else {  // for multi class classification
            createMultiClassClassifierMetrics();
            return getMacroAverage(); // in multi class case return average performance for all classes
        }
    }

    
    private ClassificationMetrics createBinaryClassifierMetrics() {
        ClassificationMetrics cm = new ClassificationMetrics(confusionMatrix);
        cm.setClassLabel(classLabels.get(1));
        // ovo ispod sve prebaci u ClassificationMetrics
        cm.set("TotalClasses", classLabels.size()); // trebalo bi da ima uvek dve klase jer je ovo binary classifier
        cm.set("TotalItems", cm.getTotal());

        cm.set(EvaluationMetrics.ACCURACY, cm.getAccuracy());
        cm.set(EvaluationMetrics.PRECISION, cm.getPrecision());
        cm.set(EvaluationMetrics.RECALL, cm.getRecall());
        cm.set(EvaluationMetrics.F1SCORE, cm.getF1Score());
        
        return cm;
    }

    // ovde negde zezne evaluaciju iza iris
    private Map<String, ClassificationMetrics> createMultiClassClassifierMetrics() {
        metricsByClass = new HashMap();
        for (int clsIdx = 1; clsIdx < classLabels.size(); clsIdx++) {
            // kreiraj metrike za svaku pojedinacnu klasu 
            ClassificationMetrics classMetrics = new ClassificationMetrics(confusionMatrix, classLabels.get(clsIdx), clsIdx); // confusionMatrix  // ne sme da uzme tp kao za binarni nego mora ovaj sa cls idex

            // ovo moze da se prebaci u konstruktor, isto i za binarnu klasifikaciju
            classMetrics.set(EvaluationMetrics.ACCURACY, classMetrics.getAccuracy());
            classMetrics.set(EvaluationMetrics.PRECISION, classMetrics.getPrecision());
            classMetrics.set(EvaluationMetrics.RECALL, classMetrics.getRecall());
            classMetrics.set(EvaluationMetrics.F1SCORE, classMetrics.getF1Score());

            metricsByClass.put(classLabels.get(clsIdx), classMetrics);
        }
        
        return metricsByClass;
    }


    /**
     * Process single classification result and update confusion matrix.
     * 
     * @param actual actual class (binary encoded)
     * @param predicted predicted class (binary encoded)
     */
    private void processResult(float[] actual, float[] predicted) {

        // binary classification 
        if (classLabels.size() == 2) { // if its a binary classifier //FIX: da li ovde treba da bude 2??? Bilo je 1
            if ((actual[0] == 1) && (predicted[0] >= threshold)) {
                confusionMatrix.inc(1, 1); // tp is at [1, 1]
            } else if ((actual[0] == 0) && (predicted[0] < threshold)) {
                confusionMatrix.inc(0, 0); // tn is at [0, 0]
            } else if ((actual[0] == 0) && (predicted[0] >= threshold)) {
                confusionMatrix.inc(0, 1); // fp is at [0, 1]
            } else if ((actual[0] == 1) && (predicted[0] < threshold)) {
                confusionMatrix.inc(1, 0); // fn is at [1, 0]
            }
        } else { // multi class classifier
            int actualIdx = indexOfMax(actual);
            int predictedIdx = indexOfMax(predicted); // ako su svi nule predictsIdx je od NEGATIVE
            // ovde negde zezne , kak oje sa none, mozda uzeti maksimalni bez thresholda, gde sa none?                       
            confusionMatrix.inc(actualIdx, predictedIdx);
        }
    }
    
    private void processResult(Tensor2D actual, Tensor2D predicted) {    
        // ovo kada je u batch modu
        
        
        // binary classification 
        if (classLabels.size() == 2) { // if its a binary classifier //FIX: da li ovde treba da bude 2??? Bilo je 1
           
            for(int c=0; c < actual.cols(); c++) {
                if ((actual.get(0, c) == 1) && (predicted.get(0, c) >= threshold)) {
                    confusionMatrix.inc(1, 1); // tp is at [1, 1]
                } else if ((actual.get(0, c) == 0) && (predicted.get(0, c) < threshold)) {
                    confusionMatrix.inc(0, 0); // tn is at [0, 0]
                } else if ((actual.get(0, c) == 0) && (predicted.get(0, c) >= threshold)) {
                    confusionMatrix.inc(0, 1); // fp is at [0, 1]
                } else if ((actual.get(0, c) == 1) && (predicted.get(0, c) < threshold)) {
                    confusionMatrix.inc(1, 0); // fn is at [1, 0]
                }
            }
        } else { // multi class classifier
            float[] actualCol = new float[actual.rows()];
            float[] predictedCol = new float[predicted.rows()];
            for(int c=0; c < actual.cols(); c++) { // iterate all items in batch
                actualCol = actual.getCol(c, actualCol);
                predictedCol = predicted.getCol(c, predictedCol);
                int actualIdx = indexOfMax(actualCol); // max u koloni daj max ali podimenziji u koloni
                int predictedIdx = indexOfMax(predictedCol); // ako su svi nule predictsIdx je od NEGATIVE
                confusionMatrix.inc(actualIdx, predictedIdx);
            }
        }        
    }

    /**
     * Returns index of max element in specified array.
     * If all elements are zero (negative example) returns 0.
     * Used only for multi class classificati case.
     *
     * @param array
     * @return index of max value
     */
    private int indexOfMax(final float[] array) {
        int maxIdx =0;
        for (int i = 1; i < array.length; i++) {
//            if (array[i] >= threshold) {
//                if (maxIdx==-1) maxIdx = i;
//                    else if (array[i] > array[maxIdx]) maxIdx = i;
//            }
              if (array[i] > array[maxIdx]) maxIdx = i;
            
        }    
        
//        if (maxIdx == -1) return 0; // ovde ne bi trebao da vracam nulu!!!        to je kao none
//            else return maxIdx+1; // +1 because zero is negative - zasto bre +1???? Zato sto sam za nultu klasu stavio negativnu za viseklasnu klasifikaciju, ali to nije dobro z abinarnu gde se ovo zapravo i ne koristi
            return maxIdx+1;
    }

    public float getThreshold() {
        return threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }

    // ali kako po klasama? ovo je za sve klase zajedno
    //  macro-average will compute the metric independently for each class and then take the averag
    public ClassificationMetrics getMacroAverage() {
        float accuracySum = 0, precisionSum = 0, recallSum = 0, f1scoreSum = 0;
        int tpSum=0, tnSum=0, fpSum=0, fnSum=0, totalSum=0;
        // ove nisu dobre class 1 ima 100 tp a 900 fp
        // medjutim ostale klase uopste nemaju tp samo fn
        // proveri generisanje classification metrika po klasama!
        for (ClassificationMetrics cm : metricsByClass.values()) {
            accuracySum += cm.getAccuracy(); // ako jedanod ovih ude NaN svi budu NaNa a bude NaN kad neke od klasa nema pa se deli sa nulom
            recallSum += cm.getRecall();
            precisionSum += cm.getPrecision();
            f1scoreSum += cm.getF1Score();
            tpSum += cm.getConfusionMatrix().getTruePositive(cm.getClassIdx()); 
            tnSum += cm.getConfusionMatrix().getTrueNegative(cm.getClassIdx());
            fpSum += cm.getConfusionMatrix().getFalsePositive(cm.getClassIdx());
            fnSum += cm.getConfusionMatrix().getFalsePositive(cm.getClassIdx());
        } 

        int count = metricsByClass.values().size(); // number of classes

        // ovde bih morao da imam ; ali confucion matrica je potpuno pogresna u ovom slucaju. Sve vrednsti roizasle iz nje nemaju smiao koji treba samo ove koje su setovane
        ClassificationMetrics total = new ClassificationMetrics(tnSum, fpSum, fnSum, tpSum);

        total.setClassLabel("Macro Average"); // za sve klase
        total.set(EvaluationMetrics.ACCURACY, accuracySum / count);
        total.set(EvaluationMetrics.PRECISION, precisionSum / count);
        total.set(EvaluationMetrics.RECALL, recallSum / count);
        total.set(EvaluationMetrics.F1SCORE, f1scoreSum / count);

        return total;
    }

    /**
     * Calculates macro average for the given list of ClassificationMetrics.
     * Used for multi-class classification.
     * Its suitable for balanced classes, but if not micro averaging is more suitable.
     * Micro averaging is summing confusion matrices for individual classes.
     *
     * @param metrics
     * @return
     */
    public static EvaluationMetrics macroAverage(Collection<EvaluationMetrics> metrics) {
        float accuracySum = 0, precisionSum = 0, recallSum = 0, f1scoreSum = 0;

        // sum all
        for (EvaluationMetrics em : metrics) {
            accuracySum += ((ClassificationMetrics)em).getAccuracy();
            recallSum += ((ClassificationMetrics)em).getRecall();
            precisionSum += ((ClassificationMetrics)em).getPrecision();
            f1scoreSum += ((ClassificationMetrics)em).getF1Score();
        }

        int count = metrics.size();

        // dicide by number of metrics
        EvaluationMetrics total = new EvaluationMetrics();
        total.set(EvaluationMetrics.ACCURACY, accuracySum / count);
        total.set(EvaluationMetrics.PRECISION, precisionSum / count);
        total.set(EvaluationMetrics.RECALL, recallSum / count);
        total.set(EvaluationMetrics.F1SCORE, f1scoreSum / count);

        return total;
    }
    
    public Map<String, ClassificationMetrics> getMetricsByClass() {
        return metricsByClass;
    }

    public ConfusionMatrix getConfusionMatrix() {
        return confusionMatrix;
    }  
    

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(System.lineSeparator()).append("------------------------------------------------------------------------").append(System.lineSeparator()).
        append("CLASSIFIER EVALUATION RESULTS ").append(System.lineSeparator()).append("------------------------------------------------------------------------").append(System.lineSeparator());
        
        sb.append("Total classes: ");
        
        if (classLabels.size() > 2) {
            sb.append(classLabels.size()-1).append(System.lineSeparator()); // za multi class smanji za -1 zbog NONE/negative
        } else {
            sb.append(classLabels.size()); // za binary classification kazi 2 klase
        }
        sb.append(System.lineSeparator());
        
        sb.append("Results by each class label").append(System.lineSeparator());

        for (String label : metricsByClass.keySet()) {
            EvaluationMetrics result = metricsByClass.get(label);
            sb.append(result).append(System.lineSeparator());
        }

        return sb.toString();
    }

}
