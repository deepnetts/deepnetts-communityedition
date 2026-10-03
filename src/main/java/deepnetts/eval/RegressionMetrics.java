package deepnetts.eval;

import java.util.HashMap;
import javax.visrec.ml.eval.EvaluationMetrics;

/**
 * Common metrics for regression models.
 */
public class RegressionMetrics extends EvaluationMetrics {
    private float maxError,
                  r2,
                  meanSquaredError,
                  rootMeanSquaredError,
                  squaredErrorSum,
                  meanAbsoluteError,
                  meanAbsolutePercentageError, 
                  residualStandardError, 
                  fStat;
    
    private static final HashMap<String, String> descriptions;
    
    static {
        descriptions = new HashMap<>();
        descriptions.put("maxError", "The biggest error in prediction by the regression model");
        descriptions.put("r2", "Proportion of variance explained by the model. Intuitively how much the model prediction is better than using the mean value as prediction. A value between 0 and 1, where 1 is the best and 0 worst ");
        descriptions.put("meanSquaredError", "Mean/average value of squared errors (the difference betwen actual and predicted value). Highly sensitive to large errors and outliers in inputs. The lower the better ideally 0.");
        descriptions.put("rootMeanSquaredError", "Squared root of the meanSquaredError. But in the same units as observerd value, makes it easier to interpret, and it is less sensitive to large error values. The lower the better ideally 0.");
        descriptions.put("squaredErrorSum", "Total sum of squared errors. The lower the better.");
        descriptions.put("meanAbsoluteError", "Average error. Less sensitive to larger errors and outliers than meanSquaredError. The lower the better ideally 0.");
        descriptions.put("meanAbsolutePercentageError", "Mean/average of the absolute errors relative to their targets.Sensitive to relative erors");
    }
    
    /**
     * Proportion of variance explained by the model.A value between 0 and 1, where 1 is the best and 0 worst.
     * Intuitively how much the model prediction is better than using mean value as prediction.
     */
    public float getR2() {
        return r2;
    }

    public void setR2(float r2) {
        this.r2 = r2;
        set(EvaluationMetrics.R_SQUARED, r2);    
        
    }

    /**
     * Mean squared error is the average value of the sum of squared errors.
     */
    public float getMeanSquaredError() {
        return meanSquaredError;
    }

    public void setMeanSquaredError(float meanSquaredError) {
        this.meanSquaredError = meanSquaredError;
        this.set(EvaluationMetrics.MEAN_SQUARED_ERROR, meanSquaredError); // for compatibility with EvaluationMetrics from visrec
    }

    public float getRootMeanSquaredError() {
        return rootMeanSquaredError;
    }

    public void setRootMeanSquaredError(float rootMeanSquaredError) {
        this.rootMeanSquaredError = rootMeanSquaredError;
        this.set(EvaluationMetrics.ROOT_MEAN_SQUARED_ERROR, rootMeanSquaredError); // for compatibility with EvaluationMetrics from visrec        
    }

    public float getMeanAbsoluteError() {
        return meanAbsoluteError;
    }

    public void setMeanAbsoluteError(float meanAbsoluteError) {
        this.meanAbsoluteError = meanAbsoluteError;
        this.set(EvaluationMetrics.MEAN_ABSOLUTE_ERROR, meanAbsoluteError); // for compatibility with EvaluationMetrics from visrec                
    }

    public float getMeanAbsolutePercentageError() {
        return meanAbsolutePercentageError;
    }

    public float getSquaredErrorSum() {
        return squaredErrorSum;
    }

    public void setSquaredErrorSum(float squaredErrorSum) {
        this.squaredErrorSum = squaredErrorSum;
    }

    public void setMeanAbsolutePercentageError(float meanAbsolutePercentageError) {
        this.meanAbsolutePercentageError = meanAbsolutePercentageError;
    }

    public float getMaxError() {
        return maxError;
    }

    public void setMaxError(float maxError) {
        this.maxError = maxError;
    }

    public float getResidualStandardError() {
        return residualStandardError;
    }

    public void setResidualStandardError(float residualStandardError) {
        this.residualStandardError = residualStandardError;
    }

    public float getFStat() {
        return fStat;
    }

    public void setFStat(float fStat) {
        this.fStat = fStat;
    }
    
    

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        
        sb.append("RegressionMetrics{").append(System.lineSeparator()).
                append("r2=").append(r2).append(" ").append(descriptions.get("r2")).append(System.lineSeparator()).
                append("meanSquaredError=").append(meanSquaredError).append(" ").append(descriptions.get("meanSquaredError")).append(System.lineSeparator()).
                append("rootMeanSquaredError=").append(rootMeanSquaredError).append(" ").append(descriptions.get("rootMeanSquaredError")).append(System.lineSeparator()).
                append("squaredErrorSum=").append(squaredErrorSum).append(" ").append(descriptions.get("squaredErrorSum")).append(System.lineSeparator()).
                append("meanAbsoluteError=").append(meanAbsoluteError).append(" ").append(descriptions.get("meanAbsoluteError")).append(System.lineSeparator()).
                append("meanAbsolutePercentageError=").append(meanAbsolutePercentageError).append(" ").append(descriptions.get("meanAbsolutePercentageError")).append(System.lineSeparator()).
                append("maxError=").append(maxError).append(" ").append(descriptions.get("maxError")).append(System.lineSeparator());
     
        return sb.toString();
    }
       
}