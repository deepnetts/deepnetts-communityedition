package deepnetts.net.train;

import deepnetts.eval.ClassificationMetrics;
import deepnetts.eval.RegressionMetrics;
import java.util.Enumeration;
import java.util.Properties;
import javax.visrec.ml.eval.EvaluationMetrics;
import static javax.visrec.ml.eval.EvaluationMetrics.ACCURACY;
import org.json.JSONObject;

/**
 * All information about the completed training including training settings, epochs, loss and evaluation metrics.
 */
public class TrainingResult {
            
    private final Properties properties; // various additional training settings / parameters used for training
    private final int epochs;
    private final float loss;
    private final EvaluationMetrics evalMetrics;

    public TrainingResult(Properties properties, int epochs, float loss, EvaluationMetrics evalMetrics) {
        this.properties = properties;
        this.epochs = epochs;
        this.loss = loss;
        this.evalMetrics = evalMetrics;
    }
        
    public EvaluationMetrics getEvaluationMetrics() {
        return evalMetrics;
    }

    public int getEpochs() {
        return epochs;
    }

    public float getLoss() {
        return loss;
    }


    public Properties getProperties() {
        return properties;
    }

    @Override
    public String toString() {
        // resi formatiranje ovoga da bude citljivo - mozda posebna metoda asJson ili asCSV
        return "TrainingResult{" + "settings=" + properties + ", epochs=" + epochs + ", loss=" + loss + ", performanceMeasure=" + evalMetrics + '}';
    }
    
    public String asJson() {
        JSONObject jsonObj = new JSONObject();
        
        jsonObj.put("epochs", epochs);
        jsonObj.put("loss", loss);
        
        // settings
        Enumeration en = properties.keys();
        while(en.hasMoreElements()) {
            String key = (String)en.nextElement();
            jsonObj.put(key, properties.get(key));
        }
        
        if (evalMetrics instanceof ClassificationMetrics) {
           // ClassificationMetrics clsMetrics = (ClassificationMetrics)evalMetrics;
            jsonObj.put("accuracy", evalMetrics.get(ACCURACY));
            jsonObj.put("precision", evalMetrics.get(EvaluationMetrics.PRECISION));
            jsonObj.put("recall", evalMetrics.get(EvaluationMetrics.RECALL));
            jsonObj.put("fscore", evalMetrics.get(EvaluationMetrics.F1SCORE));
        } else if (evalMetrics instanceof RegressionMetrics) {
            jsonObj.put("r2", evalMetrics.get(EvaluationMetrics.R_SQUARED));
            jsonObj.put("mse", evalMetrics.get(EvaluationMetrics.MEAN_SQUARED_ERROR));
            jsonObj.put("rmse", evalMetrics.get(EvaluationMetrics.ROOT_MEAN_SQUARED_ERROR));
            jsonObj.put("mae", evalMetrics.get(EvaluationMetrics.MEAN_ABSOLUTE_ERROR));
        }
                
        return jsonObj.toString();
    }
    
    
    
}
