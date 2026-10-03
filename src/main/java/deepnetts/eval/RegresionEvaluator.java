package deepnetts.eval;

import deepnetts.net.NeuralNetwork;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import javax.visrec.ml.eval.Evaluator;
import deepnetts.data.MLDataItem;
import deepnetts.tensor.Tensor1D;
import java.util.Collection;

/**
 * Evaluates regressor neural network for specified data set.
 * Assumes only one output at the moment.
 * TODO: f statistic
 * https://www.statisticshowto.com/probability-and-statistics/f-statistic-value-test/
 */
public class RegresionEvaluator implements Evaluator<NeuralNetwork, DataSet<? extends MLDataItem>> {

    private final float EPS = 0.1f; // 000001f; ako je target 0
    // radi samo za single output regresiju
    
    @Override
    public EvaluationMetrics evaluate(NeuralNetwork neuralNet, DataSet<? extends MLDataItem> testSet) {
        RegressionMetrics result = new RegressionMetrics();

        float sqrErrSum = 0;
        float absErrSum = 0;
        float totalSqrSum=0;
        float mape = 0;
        final float targetMean = mean(testSet);
        final int patternCount = testSet.size();  
        float maxError = 0;
        
        for (MLDataItem item : testSet) {
            neuralNet.setInput(item.getInput());
            final float predicted = neuralNet.getOutputAsTensor().getValues()[0];
            final float target = item.getTargetOutput().getValues()[0];
            final float error = predicted - target;
            
            maxError = Math.max(Math.abs(error), maxError);
                                
            sqrErrSum += Math.pow(error, 2); // rss - residual sum of squares
            absErrSum += Math.abs(error);      
            mape += Math.abs(error)/Math.max(EPS, Math.abs(target)); // problem je ako je target 0 zasto se to desi???
            totalSqrSum += Math.pow(target - targetMean, 2);
        }
        
        final float mse = sqrErrSum / patternCount;
        final float rmse = (float)Math.sqrt(mse);
        final float mae = absErrSum / patternCount;
        mape = mape / patternCount; // mape bude ogroman

        result.setMeanSquaredError(mse);
        result.setRootMeanSquaredError(rmse);
        result.setSquaredErrorSum(sqrErrSum);        
        result.setMeanAbsoluteError(mae);
        result.setMeanAbsolutePercentageError(mape);
        result.setMaxError(maxError);
        
        float r2 = 1 - sqrErrSum / totalSqrSum;
        result.setR2(r2); // https://en.wikipedia.org/wiki/Coefficient_of_determination koliko je gore od srednje vrednosti        

// procena standardne devijacije greske?        standard error
//        float rse = (float)Math.sqrt(sqrErrSum / (float)(patternCount - 2));
//        result.setResidualStandardError(rse);
//        result.set(EvaluationMetrics.RESIDUAL_STANDARD_ERROR, rse);

// TODO: f statistcs
//        final float fStat = ((tss-rss)/(float)numInputs) / (float)(rss / ( numItems - numInputs - 1));
//        result.set(EvaluationMetrics.F_STAT, fStat);
// https://en.wikipedia.org/wiki/F-test
 
        return result;
    }

    // works only for single output
    private float mean(DataSet<? extends MLDataItem> testSet) {
        float mean=0;
        for(MLDataItem ditem : testSet) {
            if (ditem.getTargetOutput() instanceof Tensor1D) {
                mean += ((Tensor1D)ditem.getTargetOutput()).get(0);
            } else {
                mean += ditem.getTargetOutput().mean();
            }
        }
        return mean / (float)testSet.size();
    }
    
    public static EvaluationMetrics macroAverage(Collection<EvaluationMetrics> metrics) { //  Collection<RegressionMetrics>  
        float mseSum = 0, rseSum = 0, r2Sum = 0, fstatSum = 0;

        for (EvaluationMetrics em : metrics) {
            mseSum += ((RegressionMetrics)em).getMeanSquaredError();
            r2Sum += ((RegressionMetrics)em).getR2();
            // TODO:dodoaj sve metrike iz regressionMetrics ovde
//            rseSum += ((RegressionMetrics)em).getResidualStandardError(); 
//            fstatSum += ((RegressionMetrics)em).getFStat();
        }

        int count = metrics.size();

        EvaluationMetrics total = new EvaluationMetrics();
        total.set(EvaluationMetrics.MEAN_SQUARED_ERROR, mseSum / count);
        total.set(EvaluationMetrics.R_SQUARED, r2Sum / count);
//        total.set(EvaluationMetrics.F_STAT, fstatSum / count);
//        total.set(EvaluationMetrics.RESIDUAL_STANDARD_ERROR, rseSum / count);        

        return total;
    }    

}
