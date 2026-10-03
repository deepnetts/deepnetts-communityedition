package deepnetts.eval;

import deepnetts.net.NeuralNetwork;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.data.MLDataItem;

/**
 * Utility methods for evaluating machine learning models.
 */
public class Evaluators {

    private Evaluators() { }

    /**
     * Evaluates specified neural network with test set, as a regression model and returns basic regression evauation metrics.
     * Tells you how good the network approximates function described with test set.
     * 
     * @param neuralNet
     * @param testSet
     * @see  EvaluationMetrics
     * @return regression performance measures
     */
    public static RegressionMetrics evaluateRegressor(NeuralNetwork<?> neuralNet, DataSet<? extends MLDataItem> testSet) {
        RegresionEvaluator eval = new RegresionEvaluator();
        return (RegressionMetrics) eval.evaluate(neuralNet, testSet);
    }

    /**
     *
     * @param neuralNet
     * @param testSet
     * @return classification performance measure
     */
    public static ClassificationMetrics evaluateClassifier(NeuralNetwork<?> neuralNet, DataSet<? extends MLDataItem> testSet) {
        ClassifierEvaluator eval = new ClassifierEvaluator();
         return eval.evaluate(neuralNet, testSet);
    }

}
