package deepnetts.net.train;

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.data.TabularDataSet;
import deepnetts.eval.ClassifierEvaluator;
import javax.visrec.ml.eval.Evaluator;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.NeuralNetwork;
import javax.visrec.ml.data.DataSet;
import org.apache.commons.lang3.SerializationUtils;
import deepnetts.eval.RegresionEvaluator;
import deepnetts.data.MLDataItem;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Properties;
import java.util.logging.Logger;

/**
 * Split data set into k parts of equal sizes (folds), then
 * train model with k-1 folds, and validate with remaining 1 fold. 
 * Repeat that k times each with different validation fold.
 * Commonly used technique to estimate generalization ability of the model.
 */
public class KFoldCrossValidation {
    /*
     * https://svn.code.sf.net/p/java-ml/code/trunk/src/net/sf/javaml/classification/evaluation/CrossValidation.java
     * http://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold
     */
    private int numSplits; //number of folds
    
    private NeuralNetwork neuralNetwork; // arhitektura neuronske mreze

    private DataSet<?> trainingSet;
  
    private Evaluator<NeuralNetwork, DataSet<? extends MLDataItem>> evaluator; // mogao bi u logu da ispisuje rezultate evaluacije kao json    
      
    private List<NeuralNetwork> trainedNetworks; // list of trained networks - one for each validation fold
    private List<TrainingResult> trainingResults;
    private List<EvaluationMetrics> evaluationResults;
    private EvaluationMetrics bestResult, macroAverage;
    private NeuralNetwork bestNetwork;    
    private TrainingListener trainingListener;
    private int bestIdx;    
    
      private static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());

    // TODO: set metric using builder, use default if not set
    
    // TODO: repeat koliko puta da ponovi krosvalidaciju sa ralicitim podelama data set

    // prevent instantiation using constructor, allow only using builder
    private KFoldCrossValidation() {
    }

    
    public void run() {
        // split the training set into folds
        DataSet[] folds = (DataSet[]) trainingSet.split(numSplits);
        
        // init internal structures for storing results
        trainedNetworks = new ArrayList<>();
        trainingResults = new ArrayList<>();
        evaluationResults = new ArrayList<>();
        
        // at the end this will point to best network for the specified eval metric
        bestIdx = -1;
        
        // Evaluation metric to use for comparison
        String metric;
        if (evaluator instanceof ClassifierEvaluator) {
            metric = EvaluationMetrics.F1SCORE;
        } else {
            metric = EvaluationMetrics.R_SQUARED;
        }        
               
        LOGGER.info("Running KFold Crossvalidation with "+numSplits+" folds");
        
        DataSet<? extends MLDataItem> trainingFolds=null;
        
        // create training and validation sets from folds
        for (int valFoldIdx = 0; valFoldIdx < numSplits; valFoldIdx++) { // val fold idx
            DataSet validationFold = folds[valFoldIdx]; // why only tabular data set?
            // TODO: add support for image data sets
            if (trainingSet instanceof TabularDataSet) {
                TabularDataSet tabTrainingFolds = new TabularDataSet(((TabularDataSet)trainingSet).getNumInputs(), ((TabularDataSet)trainingSet).getNumOutputs());
                tabTrainingFolds.setColumnNames(((TabularDataSet)trainingSet).getColumnNames());

                for (int trainFoldIdx = 0; trainFoldIdx < numSplits; trainFoldIdx++) {
                    if (trainFoldIdx == valFoldIdx) continue;
                    tabTrainingFolds.addAll(folds[trainFoldIdx]);
                }
                trainingFolds = tabTrainingFolds;
            } else {
                //throw new NotImplementedException("Support sor this type of data set is not yet implemented");
                ImageSet imgTrainingFolds = new ImageSet(((ImageSet)trainingSet).getImageWidth(), ((ImageSet)trainingSet).getImageHeight());
                imgTrainingFolds.setAsTargetColumns(trainingSet.getTargetColumnsNames());

                for (int trainFoldIdx = 0; trainFoldIdx < numSplits; trainFoldIdx++) {
                    if (trainFoldIdx == valFoldIdx) continue;
                    imgTrainingFolds.addAll(folds[trainFoldIdx]);                
                }
                trainingFolds = imgTrainingFolds;
            }

            // clone the original network each time before training - create a new instace that will be added to trainedNetworks
            NeuralNetwork<BackpropagationTrainer> currentNeuralNet = SerializationUtils.clone(this.neuralNetwork); // ovde bi morao traineru da prosledjuje kloniranu mrezu
            BackpropagationTrainer trainer = currentNeuralNet.getTrainer();
            if (trainingListener != null) { // add training listener only if it has been externaly set to this KFoldCrossValidation object
                trainer.addListener(trainingListener);
            }

            LOGGER.info("Training using validation fold "+(valFoldIdx+1)+" of "+numSplits);
             
            trainer.train(trainingFolds);
            trainedNetworks.add(currentNeuralNet);
            trainer.removeListener(trainingListener);
            
            EvaluationMetrics em = evaluator.evaluate(currentNeuralNet, validationFold);
            evaluationResults.add(em);
            Properties prop = new Properties();
            prop.put("fold", valFoldIdx);
            TrainingResult trainRes = new TrainingResult(prop, trainer.getCurrentEpoch(), trainer.getTrainingLoss(), em);
            trainingResults.add(trainRes);

            // uporedi tekuci rezultat sa best - proveri samo jos ovo ali deluje dobro!
            if (isBetter(em, bestResult, metric)) {
                bestIdx = trainedNetworks.indexOf(currentNeuralNet);
                bestResult = em;
            }
        }
        
        bestNetwork = trainedNetworks.get(bestIdx);
                    
        if (evaluator instanceof ClassifierEvaluator) {
            macroAverage = ClassifierEvaluator.macroAverage(evaluationResults);
        } else {
            macroAverage = RegresionEvaluator.macroAverage(evaluationResults); // 
        }
        
        // da se sve loguje i da na kraju poziva save lsitener za sve treninge sa putanjom gde da ih snima. Traineru da kaci listener/funkciju Consumer ili FUnction 
                             
    }

    public static Builder builder() {
        return new Builder();
    }

    private boolean isBetter(EvaluationMetrics em, EvaluationMetrics bestResult, String metricName) {
        if (bestResult==null) return true;
        if (em.get(metricName) > bestResult.get(metricName)) {
            return true;
        } else {
            return false;
        }
    }

    public EvaluationMetrics getBestResult() {
        return bestResult;
    }

    public EvaluationMetrics getMacroAverage() {
        return macroAverage;
    }
        
    public NeuralNetwork getBestNetwork() {
        return bestNetwork;
    }

    public List<TrainingResult> getTrainingResults() {
        return trainingResults;
    }

    public TrainingListener getTrainingListener() {
        return trainingListener;
    }

    /**
     * Builder object for KFoldCrossValidation.
     */
    public static class Builder {

        KFoldCrossValidation kFoldCV = new KFoldCrossValidation();

        public Builder numSplits(int numSplits) {
           kFoldCV.numSplits = numSplits;
           return this;
        }

        public Builder model(NeuralNetwork neuralNet) {
            kFoldCV.neuralNetwork = neuralNet;
            return this;
        }

        // training set, add test set
        public Builder trainingSet(DataSet dataSet) {
            kFoldCV.trainingSet = dataSet;
            return this;
        }          

        public Builder evaluator(Evaluator<NeuralNetwork, DataSet<? extends MLDataItem>> evaluator) {
            kFoldCV.evaluator = evaluator;
            return this;
        }
        
        public Builder trainingListener(TrainingListener trainingListener) {            
            kFoldCV.trainingListener = Objects.requireNonNull(trainingListener, "TrainingListener must not be Null!");
            return this;            
        }        
        
        public KFoldCrossValidation build() {
            return kFoldCV;
        }

    }
    
}
