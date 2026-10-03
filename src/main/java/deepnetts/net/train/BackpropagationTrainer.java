 package deepnetts.net.train;

import deepnetts.net.train.opt.OptimizerType;
import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.AbstractLayer;

import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.loss.LossFunction;
import deepnetts.util.FileIO;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Properties;
import javax.visrec.ml.data.DataSet;
import java.io.ObjectInputStream;
import java.util.LinkedList;
import deepnetts.data.MLDataItem;
import deepnetts.eval.ClassificationMetrics;
import deepnetts.net.Mode;
import deepnetts.net.layers.FlattenLayer;
import deepnetts.net.layers.MaxPoolingLayer;
import deepnetts.net.train.opt.AdaDeltaOptimizer;
import deepnetts.net.train.opt.AdamOptimizer;
import deepnetts.net.train.opt.LearningRateDecay;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.TensorBase;
import deepnetts.util.Debugger;
import deepnetts.util.ImagePreprocessing;
import java.sql.BatchUpdateException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Backpropagation training algorithm for feed forward and convolutional neural networks.
 * Backpropagation is a supervised machine learning algorithm which iteratively
 * reduces prediction error, by looking for the minimum of loss function.
 *
 * @see FeedForwardNetwork
 * @see ConvolutionalNetwork
 * @see LossFunction
 * @see OptimizerType

 */
public class BackpropagationTrainer implements Trainer, Serializable {

    /**
     * Training stops when this number of epochs is reached regardless
     * of the total network error.
     * One epoch represents one pass of the entire training set.
     */
    private long maxEpochs = 100000L;

    /**
     * Training stops once total training error has reached this value.
     */
    private float stopError = 0.01f;

    /**
     * Global learning rate, which controls the step size for the weights adjustment.
     * Effectively a percent of error used to change weight.
     */
    private float learningRate = 0.01f;
    
    private float dropout;

    /**
     * Optimization algorithm type
     */
    private OptimizerType optType = OptimizerType.SGD;

    /**
     * Global momentum parameter
     */
    private float momentum = 0.5f;

    /**
     * Set to true to use batch mode training
     */
    private boolean batchMode = false;

    /**
     * Size of mini batch. When full batch is used, this equals training set size
     */
    private int batchSize;

    /**
     * Flag to stop training
     */
    private boolean stopTraining = false;

    /**
     * Current training epoch
     */
    private int epoch;

    /**
     * Value of loss function calculated on validation set
     */
    private float valLoss=0;

    private float trainAccuracy=0, valAccuracy=0;
    private float stopAccuracy = 1;

    private float totalTrainingLoss;

    /**
     * Shuffle training set before each epoch during training.
     */
    private boolean shuffle = false;

    
    /**
     * A neural network to train.
     */
    private NeuralNetwork<?> neuralNet;

    private transient DataSet<? extends MLDataItem> trainingSet;

    private transient DataSet<? extends MLDataItem> validationSet;

    private LossFunction lossFunction;

    private boolean trainingSnapshots = false;
    private int snapshotEpochs = 5;    
    private String snapshotPath = ""; // snapshot path    
      
    /**
     * Use early stopping setting.
     */
    private boolean earlyStopping = false;

    /**
     * How many epochs for early stopping checkpoint.
     */
    private int checkpointEpochs=1;

    /**
     * Min delta between checkpoints to continue training
     */
    private float earlyStoppingMinLossChange=0.0001f;

    /**
     * How many checkpoints to wait before stopping training
     */
    private int earlyStoppingPatience = 3;
    private int earlyStoppingCheckpointCount = 0; // checkpoint counter during training

    private float prevCheckpointLoss=0;


    //private transient Evaluator<NeuralNetwork, DataSet<? extends MLDataItem>> eval = new ClassifierEvaluator();

    //regularization l1 or l2 add to loss
    private float regL2=0, regL1=0;
    
    private LinkedList<Float> lossHistory = new LinkedList<>(); // mozda linked queue duzine zadate
    
    private float[] avgConvergenceSpeeed = new float[3];


    private transient List<TrainingListener> listeners = new ArrayList<>(); // TODO: add WeakReference for all listeners

    private boolean extendedLogging = false;
    
    private static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());


    /**
     * Creates an instance of BackpropagationTrainer for the given neural network to train.
     * @param neuralNet neural network to train using this instance of backpropagation algorithm
     */
    public BackpropagationTrainer(NeuralNetwork neuralNet) {
        this.neuralNet = neuralNet;
    }

    /**
     * Creates an instance of BackpropagationTrainer with the given properties.
     * 
     * @param prop key,value pairs of properties for backpropagation
     */    
    public BackpropagationTrainer(Properties prop) {
        setProperties(prop); // all this should be done in setProperties
        // a kako ovaj setuje neuronsku mrezu sa setN
//        this.maxError = Float.parseFloat(prop.getProperty(PROP_MAX_ERROR));
//        this.maxEpochs = Integer.parseInt(prop.getProperty(PROP_MAX_EPOCHS));
//        this.learningRate = Float.parseFloat(prop.getProperty(PROP_LEARNING_RATE));
//        this.momentum = Float.parseFloat(prop.getProperty(PROP_MOMENTUM));
//        this.batchMode = Boolean.parseBoolean(prop.getProperty(PROP_BATCH_MODE));
//        this.batchSize = Integer.parseInt(prop.getProperty(PROP_BATCH_SIZE));
    }
    
    /**
     * Runs training using given training and validation sets.
     * Training set is used to train model, while validation set is used to check model evaluation metrics during the training.
     * with unseen data in order to prevent over-fitting.
     * Note that validation set is different from test set which is used after the training in order to evaluate trained model.
     * 
     * @param trainingSet set of example data to train the network
     * @param validationSet set of example data to validate the network during the training
     */
    public void train(DataSet<MLDataItem> trainingSet, DataSet<MLDataItem> validationSet) {
        this.validationSet = validationSet;
        train(trainingSet);
    }
    
    /**
     * Run training using given training set, and split part of it to use as a validation set.
     * @param trainingSet set of example data  to train the network
     * @param valSplit percent of training set to use as a validation set, value between 0 and 1, commonly something like 0.1, 0.2
     */
    public void train(DataSet<?> trainingSet, double valSplit) {
        DataSet[] trainValSets = (DataSet[]) trainingSet.split(1-valSplit, valSplit);
        this.validationSet = trainValSets[1];
        train(trainValSets[0]);
    }    

    /**
     * Runs training using specified training set.
     * Training is an iterative procedure during which network's internal parameters(weights) are adjusted in order to minimize prediction error for the given example data in training set.
     * 
     * @param trainingSet set of example data to train the network
     */
    @Override
    public void train(DataSet<? extends MLDataItem> trainingSet) {

        if (trainingSet == null) {
            throw new IllegalArgumentException("Argument trainingSet cannot be null!");
        }
        if (trainingSet.size() == 0) {
            throw new IllegalArgumentException("Training set cannot be empty!");
        }

        this.trainingSet = trainingSet;
        neuralNet.setOutputLabels(trainingSet.getTargetColumnsNames());

        int trainingSamplesCount = trainingSet.size();
        stopTraining = false;

//        if (batchMode && (batchSize == 0)) {
//            batchSize = trainingSamplesCount;
//        }
        // get batch mode and batch siz efrom the network
        if (trainingSet.get(0).getTargetOutput() instanceof Tensor2D target) {
            // neuralNet.get
            batchMode = true;
            batchSize = target.cols();
            // ali ako je izlaz isto slika 2d tensor ovo nece raditi - redefinisi nekako
        } else {
            batchMode = false;
        }
        

        neuralNet.setMode(Mode.TRAIN);
        // set properties to all layers
        for(int i=1; i<neuralNet.getLayers().size(); i++) {
            AbstractLayer layer = neuralNet.getLayers().get(i);

            if (layer instanceof MaxPoolingLayer) continue; // this type layer dont have optimizer, learning rate, etc
            
            layer.setLearningRate(learningRate);
            layer.setMomentum(momentum);
            if (regL2 != 0) layer.setL2Regularization(regL2);
            if (regL1 != 0) layer.setL1Regularization(regL1);                        
            layer.setBatchMode(batchMode);
            layer.setBatchSize(batchSize);
            layer.setOptimizerType(optType);
            if (!(layer instanceof FlattenLayer)) {
                if (optType == OptimizerType.ADADELTA )  addListener((AdaDeltaOptimizer)layer.getOptimizer());
                if (optType == OptimizerType.ADAM )  addListener((AdamOptimizer)layer.getOptimizer());
            }
                
            if (layer instanceof FullyConnectedLayer && dropout !=0) {
                ((FullyConnectedLayer)layer).setDropout(dropout);
            }
        }
        
       
        lossFunction = neuralNet.getLossFunction();

        //float[] outputError;
        TensorBase outputError;
        epoch = 0;
        totalTrainingLoss = 0;
        float prevTotalLoss = 0, totalLossChange;
        long startTraining, endTraining, trainingTime, startEpoch, endEpoch, epochTime;

        LOGGER.info("------------------------------------------------------------------------");
        LOGGER.info("TRAINING NEURAL NETWORK");
        LOGGER.info("------------------------------------------------------------------------");

        fireTrainingEvent(TrainingEvent.STARTED);
        startTraining = System.currentTimeMillis();
        
        // calculate initial loss before first iteration
        prevTotalLoss = lossFunction.valueFor(neuralNet, trainingSet);        
        LOGGER.log(Level.INFO, "Initial Train Error:"+ prevTotalLoss);
        
        do {
            epoch++;
            lossFunction.reset();
            valLoss=0;
            trainAccuracy=0;
            valAccuracy=0;

            if (shuffle) {  // maybe remove this from here, dont autoshuffle, for time series not needed - settings for Trainer
                trainingSet.shuffle(); // dataset should be shuffled before each epoch http://ruder.io/optimizing-gradient-descent/index.html#adadelta
            }
            int sampleCounter = 0;

            fireTrainingEvent(TrainingEvent.EPOCH_STARTED);
            startEpoch = System.currentTimeMillis();            

            for (MLDataItem tsItem : trainingSet) { // for all items in trainng set
                sampleCounter++;
                neuralNet.setInput(tsItem.getInput());   // set network input and automaticaly trigger  forward pass
                outputError = lossFunction.addPatternError(neuralNet.getOutputAsTensor(), tsItem.getTargetOutput()); // error for batch
                tsItem.setError(lossFunction.getPatternLoss()); // kako ovde dobiti i setovat error sa losom za tekuci item?
                neuralNet.setOutputError(outputError);
                neuralNet.backward(); // do the backward propagation using current outputError - should I use outputError as a param here?

                neuralNet.applyWeightChanges();  

                fireTrainingEvent(TrainingEvent.Type.ITERATION_FINISHED); // BATCH_FINISHED?

                if (stopTraining) break; // if training was stoped externaly by calling stop() method
            }

           if (regL2!=0) lossFunction.addRegularizationSum(regL2 * neuralNet.getL2RegSum()); // 0.00001f
                else if (regL1!=0) lossFunction.addRegularizationSum(regL1 * neuralNet.getL1RegSum());
         
            //   batch weight update after entire data set - ako vrlicina dataseta nije deljiva sa batchSize - ostatak
            if (isBatchMode() && (trainingSamplesCount % batchSize != 0)) { // full batch. zarga ovaj gore ne pokriva? @CHECK: da li je ovo dobar uslov
                neuralNet.applyWeightChanges();
            }

            endEpoch = System.currentTimeMillis();
                     
            totalTrainingLoss = lossFunction.getTotal(); // - da li total error za ceo data set ili samo za mini  batch? lossFunction.getTotalError()
            totalLossChange = totalTrainingLoss - prevTotalLoss; // todo: pamti istoriju ovoga i crtaj funkciju, to je brzina konvergencije na 10, 100, 1000 iteracija paterna - ovo treba meriti. Ovo moze i u loss funkciji
                                  
            // calculateEvaluationMetrics - mora da se napravi za batch
            trainAccuracy = calculateEvaluationMetric(this.trainingSet);  // ovo zameniti sa RMSE za regresiju i gore iznad // ako nije zadat valdiation set nemoj ni da pises?            

            if (validationSet != null) {    // kako da znam da li je klasifikacija ili regresija? mozda da imam neki setting, flag?
                valLoss = lossFunction.valueFor(neuralNet, validationSet);// lossFor(validationSet);   // pre je i test loss
                valAccuracy = calculateEvaluationMetric(validationSet);// da li ovo da radim ovde ili na event. bolje ovde zbog sinhronizacije
            }

            epochTime = endEpoch - startEpoch;
            // todo: validation error change!!! or validation  stall - decrease learning rate - create a bunch of listeners/rules that can influence training

            if (Float.isNaN(totalTrainingLoss)) {
                // TODO: revert network to previous epoch - last known working state and stop the training (opcija za snimanje mreze na svakom checkpointu )
                stopTraining = true;
                totalTrainingLoss = prevTotalLoss; // save prev loss (if NaN occured) sve somewehere this info NaN during training
                LOGGER.info("The training was interrupted due to NaN value before completing all Epochs. Epochs completed: " + epoch + "/" + maxEpochs);                
            } else {
                prevTotalLoss = totalTrainingLoss; // if it is not null set as prev loss
            }

            fireTrainingEvent(TrainingEvent.EPOCH_FINISHED);

            // EARLY STOPPING - DEBUG
            // prevCheckpointTestLoss == 0 , kolikoj evalLoss ak onema valdatio seta?
            // check for early stopping condition in each checkpointEpochs
            if (earlyStopping && (epoch > 0 && epoch % checkpointEpochs == 0)) {
                if (prevCheckpointLoss == 0) {
                    if (validationSet != null) {
                        prevCheckpointLoss = valLoss;
                    } else {
                        prevCheckpointLoss = totalTrainingLoss;
                    }
                } else {
                    // kako inicijalizovati i dodeljivati prevCheckpointLoss, ne sme biti nula jer ceodmah poceti da broji, gde i kadadodeliti pocetnu vrednost?
                    float lossChange = 0;
                    if (validationSet != null) {
                        lossChange = prevCheckpointLoss - valLoss;
                    } else {
                        lossChange = prevCheckpointLoss - totalTrainingLoss;
                    }

                    // ako je zadat validation set racunati promenu greske za validation set
                    // ako nije racunati promenu vrednosti loss-a za trening set                
                    // ne treba prevCheckpointTestLoss da bud 100 n apocetku
                    // ako je promena greske izmedju dva checkpointa previse mala ili negativna (loss raste)
                    if (lossChange < earlyStoppingMinLossChange) {
                        earlyStoppingCheckpointCount++;

                        if (earlyStoppingCheckpointCount == earlyStoppingPatience) { // ako se to desilonekoliko puta
                            stop(); // stop if test change between two checkpoints is smaller than minDeltaLoss, (that is test loss is growing, stagnating or lowering too slow)
                        }

                    } else {
                        earlyStoppingCheckpointCount = 0;  // reset counter for loss growth or stagnation
                    }

                    // save network at this checkpoint since loss if going down
                    if (validationSet != null) {
                        prevCheckpointLoss = valLoss;
                    } else {
                        prevCheckpointLoss = totalTrainingLoss;
                    }                 
                }
            }

            // make training snapshots every snapshotEpochs
            if (trainingSnapshots && (epoch > 0 && epoch % snapshotEpochs == 0)) {
                try { // save to some tmp file only if test loss was smaller
                    FileIO.writeToFile(neuralNet, snapshotPath + "_epoch_" + epoch + ".dnet"); // TODO: use constant for extension
                } catch (IOException ex) { // Catching
                    LOGGER.info(ex.getMessage());
                }                
            }
            
            lossHistory.add(totalTrainingLoss);
            if (lossHistory.size() > 10) lossHistory.removeFirst(); // loss history buffer
            
            // avg convergence speed for: 3, 5, 10 epochs           
           calculateAvgConvergenceSpeed();    
           
            if (validationSet != null)
                LOGGER.log(Level.INFO, "Epoch:"+epoch+", Time:"+epochTime+"ms, TrainError:"+totalTrainingLoss+", TrainErrorChange:"+totalLossChange+", TrainAccuracy:"+trainAccuracy+", ValError:"+valLoss+", ValAccuracy:"+valAccuracy); // dodaj train i val accuracy
           //     LOGGER.log(Level.INFO, "Epoch:{0}, Time:{1}ms, TrainError:{2}, TrainErrorChange:{3}, TrainAccuracy: {4}, ValError:{5}, ValAccuracy: {6}", new Object[]{epoch, epochTime, totalTrainingLoss, totalLossChange, trainAccuracy, valLoss, valAccuracy});// dodaj train i val accuracy
            else {
                if (extendedLogging) {
                    LOGGER.info( "Epoch:" + epoch + ", Time:" + epochTime + "ms, TrainError:" + totalTrainingLoss + ", TrainErrorChange:" + totalLossChange +
                                    ", AvgConvSpeed3: "+avgConvergenceSpeeed[0] +
                                    ", AvgConvSpeed5: "+avgConvergenceSpeeed[1] + 
                                    ", AvgConvSpeed10: "+avgConvergenceSpeeed[2] +
                                    ", TrainAccuracy: "+trainAccuracy);           
                } else {
                    LOGGER.log(Level.INFO, "Epoch:"+epoch+", Time:"+epochTime+"ms, TrainError:"+totalTrainingLoss+", TrainErrorChange:"+totalLossChange+", TrainAccuracy:"+trainAccuracy);
                    //CudaHelper.cudaGetMemInfo();
                }
            }
                
            stopTraining = stopTraining || ((epoch == maxEpochs) || (totalTrainingLoss <= stopError));    
            stopTraining =  stopTraining || (trainAccuracy >= stopAccuracy); // stop if accuracy reach 1 , more training would be overfitiing. This ,akes sense only for classification, but dont have any impact for regression
            
            //  store average convergence speed 3 5 10 epochs treba mi da poslednjih vrednosti za loss za training i validation
        } while (!stopTraining); // main training loop

        endTraining = System.currentTimeMillis();
        trainingTime = endTraining - startTraining;

        LOGGER.info(System.lineSeparator() + "TRAINING COMPLETED"); // or training interupted
        LOGGER.info("Total Training Time: " + trainingTime + "ms");
        LOGGER.info("------------------------------------------------------------------------");

        fireTrainingEvent(TrainingEvent.Type.STOPPED);
        //neuralNet.getThreadPool().shutdown(); // only for mutlithreaded ?? zasto ovo  ne bi trebalo...
        
        neuralNet.setMode(Mode.INFERENCE);
        
        // set preprocessing for network that should be used for inference
        if (trainingSet instanceof ImageSet) {
            ImagePreprocessing imgPreprocess = new ImagePreprocessing();
            ImageSet imageSet = (ImageSet)trainingSet;
            imgPreprocess.setInvertPixels(imageSet.getInvertImages());
            // imgPreprocess.setEnabled(true); ne zbog testseta
            if (imageSet.getZeroMeanPixels()) {
                imgPreprocess.setSubMean(true);
                imgPreprocess.setMean(imageSet.getMean());
            } else{
                imgPreprocess.setSubMean(false);
            }
            
            neuralNet.setPreprocessing(imgPreprocess);
        }
/*        
        if (neuralNet.getPreprocessing() != null) {
            // zasto image???
            ((ImagePreprocessing)neuralNet.getPreprocessing()).setEnabled(true);
        }        
*/        
    }

    /**
     * Returns the setting for maximum number of training epochs(iterations).
     * Training stops when the specified number of training epochs or error threshold (stopError) is reached
     * 
     * @return max training epochs
     */
    public long getMaxEpochs() {
        return maxEpochs;
    }
    
    /**
     * Sets maximum number of training epochs(iterations) for training the network.
     * Epoch is a single pass of all trainings examples from the training set.
     * The training will stop after the specified number of epochs, 
     * if the network does not reach some other stopping condition before (like error threshold).
     * 
     * @param maxEpochs the maximum number of training epochs(iterations) for training the network
     * @return this trainer
     * 
     * @see BackpropagationTrainer#stopError
     * @deprecated Use setStopEpochs instead
     */
    @Deprecated
    public BackpropagationTrainer setMaxEpochs(long maxEpochs) {
        if (maxEpochs <= 0) {
            throw new IllegalArgumentException("Max epochs should be greater then zero : " + maxEpochs);
        }
        this.maxEpochs = maxEpochs;
        return this;
    }
    /**
     * Sets number of epochs/iterations to run the training.
     * When this number of epochs is reached the training will stop, if target accuracy has not been reached before.
     * 
     * @param stopEpochs number of epochs after which training will stop
     * @return this trainer
     * 
     * @see BackpropagationTrainer#stopError
     */
    public BackpropagationTrainer setStopEpochs(long stopEpochs) {
        return setMaxEpochs(stopEpochs);
    }
    
    /**
     * Returns the setting for the stopping error threshold.
     * The training stops when total network error reaches this value.
     * @return stop error threshold
     */
    public float getMaxError() {
        return stopError;
    }
    
    /**
     * Alias for getMaxError().
     * @return 
     */
    public float getStopError() {
        return stopError;
    }

    /**
     * Sets stopping error threshold for this training.
     * The training will stop when/if training error reach this value.
     * This method will be deprecated and setStopError method should be used instead, as more intuitive.
     * 
     * @param maxError maximum error threshold
     * @return this trainer
     * @deprecated Use setStopError instead
     */
    @Deprecated
    public BackpropagationTrainer setMaxError(float maxError) {
        setStopError(maxError);
        return this;
    }
    
    
    /**
     * The training stops when/if training error reach this value.
     * 
     * @param stopError value of training error to stop the training
     * @return this trainer
     */    
    public BackpropagationTrainer setStopError(float stopError) {
        if (stopError < 0) {
            throw new IllegalArgumentException("Stop error cannot be negative: " + stopError);
        }

        this.stopError = stopError;
        return this;
    }    

    public float getStopAccuracy() {
        return stopAccuracy;
    }

    public BackpropagationTrainer setStopAccuracy(float stopAccuracy) {
        this.stopAccuracy = stopAccuracy;
        return this;
    }
    
    

    /**
     * Learning rate controls the step size as a percent of the error to use
     * for adjusting internal parameters(weights) of the neural network.
     * With too large values training may cannot be completed and error will grow, 
     * while with too small values training might last too long or get stuck in local minimum.
     * Commonly used default value for this setting is 0.01, which practically means that 1% of the error will be used for weight modification.
     * @param learningRate a value in range (0, 1), where 0.01 is being used as a default initial value
     * @return this trainer
     */
    public BackpropagationTrainer setLearningRate(float learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate cannot be negative or zero : " + learningRate);
        }
        if (learningRate > 1) {
            throw new IllegalArgumentException("Learning rate cannot be greater then 1 : " + learningRate);
        }

        this.learningRate = learningRate;
       
        return this;
    }

    /**
     * Learning rate controls the step size as a percent of the error to use
     * for adjusting internal parameters(weights) of the neural network.
     * With too large values training may cannot be completed and error will grow, 
     * while with too small values training might last too long or get stuck in local minimum.
     * Commonly used default value for this setting is 0.01, which practically means that 1% of the error will be used for weight modification.
     * @return 
     */    
    public float getLearningRate() {
        return learningRate;
    }    

    /**
     * Returns a neural network trained by this trainer.
     * @return instance of a neural network trained by this trainer
     */
    public NeuralNetwork<?> getNeuralNetwork() {
        return neuralNet;
    }
    
    /**
     * Updates learning rate for all layers during the learning rate decay.
     * Used by LearningRateDecay technique.
     * 
     * @param learningRate a value of learning rate to set for all layers
     * @see LearningRateDecay
     */
    public void updateLearningRate(float learningRate) {
        this.learningRate = learningRate;
        for(int i=1; i<neuralNet.getLayers().size(); i++) { // ovo treba da radi i na pocetku treninga i tokom treninga kad ga lr decay menja
            AbstractLayer layer = neuralNet.getLayers().get(i);            
            if (layer instanceof MaxPoolingLayer) continue; // this type layer dont have optimizer, learning rate, etc            
            layer.setLearningRate(learningRate);
            layer.getOptimizer().setLearningRate(learningRate);
        }
    }
    
    /**
     * Learning rate decay lowers the learning rate with each epoch by devayRate factor,
     * which may improve error lowering the error.
     * 
     * @param decayRate
     * @return this trainer
     */
    public BackpropagationTrainer setLearningRateDecay(float decayRate) {
        LearningRateDecay lrd = new LearningRateDecay(learningRate, decayRate);
        this.addListener(lrd);

        return this;
    }    

    /**
     * L2 regularization (sum of squares) is used to prevent overfitting and too large weights.
     * @param regL2 coefficient for L2 regularization
     * @return this trainer
     */
    public BackpropagationTrainer setL2Regularization(float regL2) {
        this.regL2 = regL2;
        return this;
    }

    /**
     * L1 regularization (sum of abs values) is used to prevent overfitting and too large weights.
     * @param regL1 coefficient for L1 regularization
     * @return this trainer
     */
    public BackpropagationTrainer setL1Regularization(float regL1) {
        this.regL1 = regL1;
        return this;
    }

    
    /**
     * Returns shuffle flag which determines if training set should be shuffled before each epoch.
     * @return value of the shuffle flag
     */
    public boolean getShuffle() {
        return shuffle;
    }

    /**
     * Sets shuffle flag which determines if training set should be shuffled before each epoch.
     * @param shuffle 
     * @return this trainer
     */
    public BackpropagationTrainer setShuffle(boolean shuffle) {
        this.shuffle = shuffle;
        return this;
    }

    /**
     * Notifies all listeners about training event.
     * @param type type of the training event
     * @see TrainingEvent
     */
    private void fireTrainingEvent(TrainingEvent.Type type) {
        for (TrainingListener l : listeners) {
            l.handleEvent(new TrainingEvent(this, type));
        }
//        if (type == TrainingEvent.Type.STOPPED) {
//            listeners.clear(); // remove alllisteners if training has stopped
//        }        
    }

    /**
     * Adds training listener to this trainer.
     * @param listener object that listens for the events in this trainer
     */
    public void addListener(TrainingListener listener) {
        Objects.requireNonNull(listener, "Training listener cannot be null!");

        synchronized(listeners) {
            if (!listeners.contains(listener)) {
                listeners.add(listener);
            }
        }
    }

    /**
     * Removes training listener from this trainer.
     * @param listener listener to remove 
     */
    public synchronized void removeListener(TrainingListener listener) {
        synchronized(listeners) {        
            listeners.remove(listener);
        }
    }
    
    public synchronized void removeAllListeners() {
        synchronized(listeners) {        
            listeners.clear();
        }
    }    

    /**
     * In batch mode weights are adjusted after the pass of all examples from the training set,
     * while in online mode weights are adjusted after each training example.
     * @see BackpropagationTrainer#setBatchMode(boolean) 
     */
    public boolean isBatchMode() {
        return batchMode;
    }

    /**
     * Sets flag whether to use batch mode during the training.
     * In batch mode weights are adjusted after the pass of all examples from the training set,
     * while in online mode weights are adjusted after each training example.
     * @param batchMode
     * @return this trainer
     */
    public BackpropagationTrainer setBatchMode(boolean batchMode) {
        this.batchMode = batchMode;
        return this;
    }

    /**
     * Batch size is number of training examples after which network's weights are adjusted.
     * @return 
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Batch size is number of training examples after which network's weights are adjusted.
     * @param batchSize
     * @return 
     */
    public BackpropagationTrainer setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    /**
     * Momentum settings helps to avoid oscillations in weight changes and get more stable and faster training.
     * It has effect only if momentum optimizer is used.
     * @param momentum a decimal value greater than zero and less than one
     * @return 
     */
    public BackpropagationTrainer setMomentum(float momentum) {
        this.momentum = momentum;
        return this;
    }

    /**
     * Momentum settings helps to avoid oscillations in weight changes and get more stable and faster training.
     * It has effect only if momentum optimizer is used.
     * @return 
     */
    public float getMomentum() {
        return momentum;
    }

    /**
     * Stops the training.
     */
    public void stop() {
        stopTraining = true;
    }

    /**
     * Total training error/loss at the current epoch.
     * The error is calculated using loss function and is referred to also as a loss.
     * @return total training error/loss at the current epoch.
     */
    public float getTrainingLoss() {
        return totalTrainingLoss;
    }

    
    /**
     * Validation loss is an error calculated using validation set, used to prevent overfitting, and validate architecture and training settings.
     * @return error/loss calculated usng validation set
     */
    public float getValidationLoss() {
        return valLoss;
    }

    /**
     * Accuracy metric which tells us a percent of correct predictions for training set.
     * @return classification accuracy for the training examples
     */
    public float getTrainingAccuracy() {
        return trainAccuracy;
    }

    /**
     * Accuracy metric which tells us a percent of correct predictions for validation set.
     * @return classification accuracy for examples in validation set
     */
    public float getValidationAccuracy() {
        return valAccuracy;
    }
    
    /** 
     * Returns the current training epoch(iteration) of this trainer.
     * Epoch is one pass of all examples from a training set. 
     * @return current training epoch
     */
    public int getCurrentEpoch() {
        return epoch;
    }

    
    public OptimizerType getOptimizer() {
        return optType;
    }

    public BackpropagationTrainer setOptimizer(OptimizerType optimizer) {
        this.optType = optimizer;
        return this;
    }

    /**
     * Test set is used after the training to estimate performance of the trained model and generalization ability with new data.
     * Examples (data) from test set should never be used during the training.
     * Tests set is commonly generated by splitting all available data in training and test sets in some ratio. 
     * @return test set - example data not used during the training, that will be used for evaluation/testing of the trained model
     */
    public DataSet<?> getTestSet() {
        return validationSet;
    }

    /**
     * Test set is used after the training to estimate performance of the trained model and generalization ability with new data.
     * Examples (data) from test set should never be used during the training.
     * Tests set is commonly generated by splitting all available data in training and test sets in some ratio. 
     * @param testSet example data not used during the training, that will be used for evaluation/testing of the trained model
     */
    public void setTestSet(DataSet<MLDataItem> testSet) {
        this.validationSet = testSet;
    }

    /**
     * Early stopping stops training if it starts converging slow, and prevents overfitting.
     * @return 
     */
    public boolean getEarlyStopping() {
        return earlyStopping;
    }

    /**
     * Early stopping stops training if it starts converging slow, and prevents overfitting.
     * @param earlyStopping
     * @return this trainer
     */
    public BackpropagationTrainer setEarlyStopping(boolean earlyStopping) {
        this.earlyStopping = earlyStopping;
        return this;
    }

    /**
     * Path to use for making snapshots - saving the current state of trained network during
     * the training in order to be able to restore it from a training point.
     * @param snapshotPath
     * @return this trainer
     */
    public BackpropagationTrainer setSnapshotPath(String snapshotPath) {
        this.snapshotPath = snapshotPath;
        return this;
    }

    /**
     * Path to use for making snapshots - saving the current state of trained network during
     * the training in order to be able to restore it from a training point if needed.
     * @return directory to store snapshots of the neural networks during the training
     */
    public String getSnapshotPath() {
        return snapshotPath;
    }        
    
    /**
     * On how many epochs to make training snapshots.
     * @return 
     */
    public int getSnapshotEpochs() {
        return snapshotEpochs;
    }

    /**
     * On how many epochs to make training snapshots.
     * @param snapshotEpochs 
     * @return
     */
    public BackpropagationTrainer setSnapshotEpochs(int snapshotEpochs) {
        this.snapshotEpochs = snapshotEpochs;
        return this;
    }
        
    /**
     * Training snapshots save the current state of the trained neural network during
     * the training in order to be able to restore it from a training point if needed.
     * @param trainingSnapshots 
     */
    public BackpropagationTrainer setTrainingSnapshots(boolean trainingSnapshots) {
        this.trainingSnapshots = trainingSnapshots;
        return this;
    }     
    
    /**
     * Returns true if network creates training snapshots, false otherwise.
     * Training snapshots save the current state of the trained neural network during
     * the training in order to be able to restore it from a training point if needed.
     * @return 
     */
    public boolean createsTrainingSnaphots() {
        return trainingSnapshots;
    }

    /**
     * Early stopping stops training if the error/loss start converging to slow.
     * If the loss change is lower than given value for patience epochs the training will stop.
     * @return 
     */
    public float getEarlyStoppingMinLossChange() {
        return earlyStoppingMinLossChange;
    }

    /**
     * Early stopping stops training if the error/loss start converging to slow.
     * If the loss change is lower than given value for patience epochs the training will stop.
     * 
     * @param earlyStoppingMinLossChange
     * @return this trainer
     */
    public BackpropagationTrainer setEarlyStoppingMinLossChange(float earlyStoppingMinLossChange) {
        this.earlyStoppingMinLossChange = earlyStoppingMinLossChange;
        return this;
    }

    /**
     * How many epochs to wait to see if the loss is lowering to slow.
     * @return 
     */
    public int getEarlyStoppingPatience() {
        return earlyStoppingPatience;
    }

    /**
     * How many epochs to wait to see if the loss is lowering to slow.
     * @param earlyStoppingPatience
     * @return 
     */
    public BackpropagationTrainer setEarlyStoppingPatience(int earlyStoppingPatience) {
        this.earlyStoppingPatience = earlyStoppingPatience;
        return this;
    }
    
    /**
     * On how many epochs the snapshots of the trained network should be created.
     * @return 
     */
    public int getCheckpointEpochs() {
        return checkpointEpochs;
    }

    /**
     * On how many epochs the snapshots of the trained network should be created.
     * @param checkpointEpochs
     * @return 
     */
    public BackpropagationTrainer setCheckpointEpochs(int checkpointEpochs) {
        this.checkpointEpochs = checkpointEpochs;
        return this;
    }    

    /**
     * Sets properties from available keys in specified prop object.
     *
     * @param prop
     */
    public final void setProperties(Properties prop) {
        if (prop.getProperty(PROP_MAX_ERROR) != null) this.stopError = Float.parseFloat(prop.getProperty(PROP_MAX_ERROR));
        if (prop.getProperty(PROP_MAX_EPOCHS) != null) this.maxEpochs = Integer.parseInt(prop.getProperty(PROP_MAX_EPOCHS));
        if (prop.getProperty(PROP_LEARNING_RATE) != null) this.learningRate = Float.parseFloat(prop.getProperty(PROP_LEARNING_RATE));
        if (prop.getProperty(PROP_MOMENTUM) != null) this.momentum = Float.parseFloat(prop.getProperty(PROP_MOMENTUM));
        if (prop.getProperty(PROP_BATCH_MODE) != null) this.batchMode = Boolean.parseBoolean(prop.getProperty(PROP_BATCH_MODE));
        if (prop.getProperty(PROP_BATCH_SIZE) != null) this.batchSize = Integer.parseInt(prop.getProperty(PROP_BATCH_SIZE));
        if (prop.getProperty(PROP_OPTIMIZER_TYPE) != null) this.optType = OptimizerType.valueOf(prop.getProperty(PROP_OPTIMIZER_TYPE));

        // iterate properties keys?use reflection to set them?
        // see kevins solution in visrec builder
    }

    /**
     * Calculates basic evaluation metric for the given validation set.
     * For classification problems calculates accuracy, while for regression problems r squared.
     * @param validationSet
     * @return 
     */
    private float calculateEvaluationMetric(DataSet<? extends MLDataItem> validationSet) {        
        EvaluationMetrics em = neuralNet.test(validationSet);
        
        if (em instanceof ClassificationMetrics)
            return em.get(EvaluationMetrics.ACCURACY); // da moze d setuje i izbaere i druge metrike tipa f1 ili sl
        else
            return em.get(EvaluationMetrics.R_SQUARED); // sta ovde staviti?
    }

    /**
     * Dropout is a technique to prevent overfitting, which skips adjusting weights for some neurons with given probability.
     * @param dropout value between 0.2 and 0.8 which represents probability  to skip adjusting weights
     * @return this trainer
     */
    public BackpropagationTrainer setDropout(float dropout) {
        this.dropout = dropout;
        return this;
    }

    /**
     * Dropout is a technique to prevent overfitting, which skips adjusting weights for some neurons with given probability.
     * @return value between 0.2 and 0.8 which represents probability  to skip adjusting weights
     */
    public float getDropout() {
        return dropout;
    }

    /**
     * This method needs to be overridden to initialize transient fields after deserialization.
     * @param ois
     * @throws ClassNotFoundException
     * @throws IOException 
     */
    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {       
        ois.defaultReadObject();
        listeners = new ArrayList<>(); 
        //eval = new ClassifierEvaluator(); // FIX: might be also regression evalutor also, how to know?
    }        
    
    /**
     * Calculates average covergence speed for last 3, 5 and 10 epochs
     */
    private void calculateAvgConvergenceSpeed() {
        if (lossHistory.size() < 10) {
            int size = lossHistory.size();
            if (size>=3) avgConvergenceSpeeed[0] = (lossHistory.get(size-1) - lossHistory.get(size-3)) / 3.0f;
            if (size>=5) avgConvergenceSpeeed[1] = (lossHistory.get(size-1) - lossHistory.get(size-5)) / 5.0f;        
            // cim dodje do da izracunava ga u drugoj if grani
        } else {        
            avgConvergenceSpeeed[0] = (lossHistory.get(9) - lossHistory.get(7)) / 3.0f;
            avgConvergenceSpeeed[1] = (lossHistory.get(9) - lossHistory.get(5)) / 5.0f;        
            avgConvergenceSpeeed[2] = (lossHistory.get(9) - lossHistory.get(0)) / 10.0f;       
        }
    }

    /**
     * Extended logging includes additional info for debugging the training.
     * @return 
     */
    public boolean getExtendedLogging() {
        return extendedLogging;
    }

    /**
     * Extended logging includes additional info for debugging the training.
     * 
     * @param extendedLogging 
     */
    public void setExtendedLogging(boolean extendedLogging) {
        this.extendedLogging = extendedLogging;
    }
    
    // move this to interface XXXConstants ?
    
    /**
     * Name of the maxError property
     */
    public static final String PROP_MAX_ERROR = "maxError";
    
    /**
     * Name of the maxEpochs property
     */
    public static final String PROP_MAX_EPOCHS = "maxEpochs";
    
    /**
     * Name of the learningRate property
     */    
    public static final String PROP_LEARNING_RATE = "learningRate";
    
    /**
     * Name of the momentum property
     */    
    public static final String PROP_MOMENTUM = "momentum";
    
    /**
     * Name of the batchMode property
     */    
    public static final String PROP_BATCH_MODE = "batchMode";
    
    /**
     * Name of the batchSize property
     */    
    public static final String PROP_BATCH_SIZE = "batchSize";     
    
    /**
     * Name of the optimizer property
     */
    public static final String PROP_OPTIMIZER_TYPE = "optimizer";  // for mini batch

}
