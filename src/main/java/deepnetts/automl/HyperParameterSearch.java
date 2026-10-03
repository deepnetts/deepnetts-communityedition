package deepnetts.automl;

import static deepnetts.automl.Parameters.HIDDEN_LAYERS;
import deepnetts.net.train.TrainingResult;
import deepnetts.core.DeepNetts;
import deepnetts.data.MLDataItem;
import deepnetts.net.FeedForwardNetwork;
import javax.visrec.ml.data.DataSet;

import deepnetts.net.NeuralNetwork;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.TrainingListener;
import deepnetts.util.TypedProperties;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.logging.Logger;
import javax.visrec.ml.eval.EvaluationMetrics;
import javax.visrec.ml.eval.Evaluator;


/**
 * The core automl class that performs automated model building and evaluation with specified parameters.
 * 
 */
public class HyperParameterSearch {

    private final Parameters params = new Parameters();
    private NeuralNetwork network; // Builder<FeedFOrwardNeuralNetwork>
    private DataSet<? extends MLDataItem> trainingSet, testSet; // maybe evaluate using validation set? or its the same

    private Evaluator<NeuralNetwork, DataSet<? extends MLDataItem>> evaluator;
    private List<Parameters.ParameterCombination> searchSpace;
    private List<TrainingResult> results = new ArrayList<>();

    private Random randomGen;

    // instead of strategy allow extending: grid, random and custom
    public static final String GRID = "grid";
    public static final String RANDOM = "random";
    public static final String DIVIDE_AND_CONQUER = "divide_and_conquer";

    private long randomSeed;
    private NetworkFactory<?> networkFactory;
    private TrainingListener trainingListener;
    
    private static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());

    public HyperParameterSearch() {

    }

    public HyperParameterSearch trainingSet(DataSet trainingSet) {
        this.trainingSet = trainingSet;
        return this;
    }
    
    public HyperParameterSearch testSet(DataSet testSet) {
        this.testSet = testSet;
        return this;
    }    
    
    public HyperParameterSearch evaluator(Evaluator<NeuralNetwork, DataSet<? extends MLDataItem>> evaluator) {
        this.evaluator = evaluator;
        return this;
    }
    
    public HyperParameterSearch trainingListener(TrainingListener trainingListener) {
        this.trainingListener = trainingListener;
        return this;
    }    
   
    private Float[] divNconqValuesForFloatRange(Float min, Float max) {
        List<Float> values = new LinkedList<>();
        divideAndConquerFloatRange(min, max, values);
        return values.toArray(new Float[0]);
    }
    
    private void divideAndConquerFloatRange(Float min, Float max, List<Float> values) {        
        if (max-min < min) return;
        
        Float mid = min + (max-min)/2;
        values.add(mid);
        
        divideAndConquerFloatRange(min, mid, values);
        divideAndConquerFloatRange(mid, max, values);
    }    
    
    private Integer[] divNconqValuesForIntRange(Integer min, Integer max) {
        List<Integer> values = new LinkedList<>();
        divideAndConquerIntRange(min, max, values);
        return values.toArray(new Integer[0]);
    }
    
    private void divideAndConquerIntRange(Integer min, Integer max, List<Integer> values) {
        
        if (max-min <5) return;
        
        Integer mid = min + (max-min)/2;
        values.add(mid);
        
        divideAndConquerIntRange(min, mid, values);
        divideAndConquerIntRange(mid, max, values);
    }
    
    
    /**
     * Returns an array of float values given the specified range and step.
     * 
     * @param min
     * @param max
     * @param step
     * @return 
     */
    private Float[] valuesForFloatRange(Float min, Float max, Float step) {
            int length =(int)((max-min)/step) + 1;
            
            Float[] values = new Float[length];
            for(int i=0; i<values.length; i++ ) {
                values[i] = min + i * step;
            }   
            
            return values;
    }
    
    private Integer[] valuesForIntRange(Integer min, Integer max, Integer step) {
            int length = (max-min)/step + 1;
            
            Integer[] values = new Integer[length];
            for(int i=0; i<values.length; i++ ) {
                values[i] = min + i * step;
            }       
            
            return values;
    }    
    
    private Integer[] randomValuesForIntRange(Integer min, Integer max, int num) {
            Integer[] values = new Integer[num];    //podeli ovo kako treba i dodaj 1 ako je nedeljivo
            for(int i=0; i<values.length; i++ ) {
                values[i] = min + randomGen.nextInt(max-min);
            }       
            
            return values;
    }    
    
    private Float[] randomValuesForFloatRange(Float min, Float max, int num) {
            Float[] values = new Float[num];    //podeli ovo kako treba i dodaj 1 ako je nedeljivo
            for(int i=0; i<values.length; i++ ) {
                values[i] = min + randomGen.nextFloat() * (max-min);
            }       
            
            return values;
    }        

    /**
     * Set searvh param with specific value.
     * 
     * @param <T>
     * @param name
     * @param value
     * @return 
     */
    public <T> HyperParameterSearch paramValue(String name, T value) {        
        params.add(name, value);
        return this;
    }    
    
    public <T> HyperParameterSearch paramValues(String name, List<T> values) {        
        params.add(name, values);
        return this;
    }

    
    // u sub clasama definisi kako ce da uzorkuje vrednosti iz opsega
    // todo - remove strategy param, and put it somewhere else in subclases of somethong, get rid of conditional logic
    public <T> HyperParameterSearch paramRange(String name, Range<T> range) {        
          randomGen = new Random(randomSeed);// ne uvek...
        String searchStrategy = GRID;
        // integer grid
        if ((range.getMin() instanceof Integer) && (range.getMax() instanceof Integer) && (range.getStep() instanceof Integer)){            
            if (searchStrategy.equals(GRID)) {
                Integer[] values = valuesForIntRange((Integer)range.getMin(), (Integer)range.getMax(), (Integer)range.getStep());
                params.add(name, values);
            } else if (searchStrategy.equals(RANDOM)) { // seed num values
                int num = 5;
                Integer[] values = randomValuesForIntRange((Integer)range.getMin(), (Integer)range.getMax(), num);
                params.add(name, values);
            } else if (searchStrategy.equals(DIVIDE_AND_CONQUER)) {
                Integer[] values = divNconqValuesForIntRange((Integer)range.getMin(), (Integer)range.getMax());
                params.add(name, values);                
            }
        } else if ((range.getMin() instanceof Float) && (range.getMax() instanceof Float) && (range.getStep() instanceof Float)){ 
          // float grid
            // ovaj preskoci verovatno zbog stepa
            if (searchStrategy.equals(GRID)) {
              Float[] values = valuesForFloatRange((Float)range.getMin(), (Float)range.getMax(), (Float)range.getStep());            
               params.add(name, values);
            } else if (searchStrategy.equals(RANDOM)) { // seed num values
                int num = range.getRandomNum();
                Float[] values = randomValuesForFloatRange((Float)range.getMin(), (Float)range.getMax(), num);    
                params.add(name, values);
            } else if (searchStrategy.equals(DIVIDE_AND_CONQUER)) {
                Float[] values = divNconqValuesForFloatRange((Float)range.getMin(), (Float)range.getMax());  
                params.add(name, values);                
            }            
        }        

        return this;
    }
    
    public Parameters getParameters() {
        return params;
    }

    public final NeuralNetwork getNetwork() {
        return network;
    }

    public final DataSet<?> getTrainingSet() {
        return trainingSet;
    }

    public final Evaluator<NeuralNetwork, DataSet<? extends MLDataItem>> getEvaluator() {
        return evaluator;
    }
    
    

    public void run() {      
        searchSpace = params.createSearchSpace();
        
        //printSearchSpace();
                   
        for (Parameters.ParameterCombination paramComb : searchSpace) {   // for each parameter combination
            Properties prop = paramComb.getAsProperties();
                                
            // create hidden layers param for factory
            int[] hiddenLayersArr = generateHiddenLayers(prop); // generisi ovde hidden layers i posalji ga factoriju, tj. ubaci u prop
            prop.put("hiddenLayersArr", hiddenLayersArr);
            
            // ovde iteriraj hiddenLayers i za svaki pozovi factory, da li se to moze prebaciti u search space?
            
            // create a new instance of neural network with specified properties
            FeedForwardNetwork neuralNet = (FeedForwardNetwork) networkFactory.createNeuralNetwork(prop);
            // ovo mora da bude kongigurabilno da moze i za konvlucione

            // set training parameters
            BackpropagationTrainer trainer = neuralNet.getTrainer();
            if (trainingListener !=null) {
                trainer.addListener(trainingListener);
            }
            
            LOGGER.info("Param combination:" + paramComb);
            trainer.train(trainingSet);
            
            if (trainingListener !=null) {
                trainer.removeListener(trainingListener);
            }
                                
            EvaluationMetrics em = evaluator.evaluate(neuralNet, testSet);                       
            TrainingResult result = new TrainingResult(prop, trainer.getCurrentEpoch(), trainer.getTrainingLoss(), em);
            
            results.add(result); // ovde cuvam rezultate svih treninga na kraju bi trebalo vratiti i statistiku pored konkretnih vrednoosti
        }      
        //LOGGER.info(results); // a mogu i da ih ispisem pomocu loggera
    }

    public List<Parameters.ParameterCombination> getSearchSpace() {
        return searchSpace;
    }

    public List<TrainingResult> getResults() {
        return results;
    }
    // dodaj getBestModel()  u odnosu na zadaru metriku
    
    public HyperParameterSearch randomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
        return this;
    }

    public HyperParameterSearch networkFactory(NetworkFactory<?> networkFactory) {
        this.networkFactory = networkFactory;    
        return this;
    }

    private int[] generateHiddenLayers(Properties params) {
        TypedProperties tprop = new TypedProperties(params);        
        int hiddenLayers=0; // ovo je ako ima jedan skriveni sloj sa zadatim brojem neurona. ako je ovo iznad jedna vrednost onda i ovo max ne trebada bud etu vec da bude lista vrednostu param
           
        if (tprop.containsKey(HIDDEN_LAYERS) )
            hiddenLayers = tprop.getInt(HIDDEN_LAYERS);        
        
        int[] hiddenLayersArr = new int[hiddenLayers];
        
        for(int i=0; i<hiddenLayers; i++) {
            if (tprop.containsKey("hiddenLayer_"+(i+1)))
                hiddenLayersArr[i] = tprop.getInt("hiddenLayer_"+(i+1));                    
        }               
        
        return hiddenLayersArr;
    }

    private void printSearchSpace() {
        // List<Parameters.ParameterCombination> searchSpace
        for(Parameters.ParameterCombination combination : searchSpace) {
            System.out.println(combination);
        }
    }
    
}
