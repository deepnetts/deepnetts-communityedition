/**
 * DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 * based learning and image recognition.
 * <p>
 * Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 * <p>
 * This file is part of DeepNetts.
 * <p>
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * <p>
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */

package deepnetts.net.train;

import deepnetts.core.DeepNetts;
import deepnetts.data.MLDataItem;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.loss.LossFunction;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.FileIO;
import org.apache.logging.log4j.LogManager;

import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import javax.visrec.ml.eval.Evaluator;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Properties;

/**
 * Backpropagation training algorithm for Feed Forward and Convolutional Neural Networks.
 * Backpropagation is a supervised machine learning algorithm which iteratively
 * reduces prediction error, by looking for the minimum of loss function.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 * @see FeedForwardNetwork
 * @see ConvolutionalNetwork
 * @see LossFunction
 * @see OptimizerType
 */
public class BackpropagationTrainer implements Trainer, Serializable {

	
    private static final long serialVersionUID = -2074106047858300199L;
	
	
    /**
     * Maximum training epochs. Training will stop when this number of epochs is
     * reached regardless the total network error.
     * One epoch represents one pass of the entire training set.
     */
    private long maxEpochs = 100000L;

    /**
     * Maximum allowed error. Training will stop once total error has reached
     * this value .
     */
    private float maxError = 0.01f;

    /**
     * Global learning rate
     */
    private float learningRate = 0.01f;

    /**
     * Optimization algorithm type
     */
    private OptimizerType optType = OptimizerType.SGD;

    /**
     * Global momentum parameter
     */
    private float momentum = 0;

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
    private float valLoss = 0, prevValLoss = 0;

    private float trainAccuracy = 0, valAccuracy = 0;

    private float totalTrainingLoss;

    /**
     * Shuffle training set before each epoch during training
     */
    private boolean shuffle = false;

    /**
     * Neural network to train
     */
    private NeuralNetwork<?> neuralNet;

    private transient DataSet<? extends MLDataItem> trainingSet;

    private transient DataSet<? extends MLDataItem> validationSet;

    /**
     * Loss function used to calculate total network error
     */
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
    private int checkpointEpochs = 1;

    /**
     * Min delta between checkpoints to continue training
     */
    private float earlyStoppingMinDelta = 0.000001f;

    /**
     * How many checkpoints to wait before stopping training
     */
    private int earlyStoppingPatience = 2;
    private int earlyStoppingCheckpointCount = 0; // checkpoint counter during training

    private float prevCheckpointTestLoss = 100f;


    private transient Evaluator<NeuralNetwork, DataSet<? extends MLDataItem>> eval = new ClassifierEvaluator();

    //regularization l1 or l2 add to loss
    private float regL2, regL1;


    private transient List<TrainingListener> listeners = new ArrayList<>(); // TODO: add WeakReference for all listeners

    private static final org.apache.logging.log4j.Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());


    /**
     * Creates and instance of Backpropagation Trainer for the specified neural network.
     *
     * @param neuralNet neural network to train using this instance of backpropagation algorithm
     */
    public BackpropagationTrainer(NeuralNetwork neuralNet) {
        this.neuralNet = neuralNet;
    }

    public BackpropagationTrainer(Properties prop) {
        // setProperties(prop); // all this should be done in setProperties
        this.maxError = Float.parseFloat(prop.getProperty(PROP_MAX_ERROR));
        this.maxEpochs = Integer.parseInt(prop.getProperty(PROP_MAX_EPOCHS));
        this.learningRate = Float.parseFloat(prop.getProperty(PROP_LEARNING_RATE));
        this.momentum = Float.parseFloat(prop.getProperty(PROP_MOMENTUM));
        this.batchMode = Boolean.parseBoolean(prop.getProperty(PROP_BATCH_MODE));
        this.batchSize = Integer.parseInt(prop.getProperty(PROP_BATCH_SIZE));
    }


    /**
     * Run training using specified training and validation sets.
     * Training set is used to train model, while validation set is used to check model accuracy with unseen data
     * during training in order to prevent overfitting.
     *
     * @param trainingSet
     * @param validationSet
     */
    public void train(DataSet<MLDataItem> trainingSet, DataSet<MLDataItem> validationSet) {
        this.validationSet = validationSet;
        train(trainingSet);
    }

    public void train(DataSet<?> trainingSet, double valPart) {
        DataSet[] trainValSets = (DataSet[]) trainingSet.split(1 - valPart, valPart); // ali da moze i jedan parametar
        this.validationSet = trainValSets[1];
        train(trainValSets[0]);
    }


    /**
     * Run training using specified training set.
     * <p>
     * Make this pure function so it can run in multithreaded - can train
     * several nn in parallel put network as param
     *
     * @param trainingSet training data to build the model
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

        if (batchMode && (batchSize == 0)) {
            batchSize = trainingSamplesCount;
        }

        // set same lr to all layers!
        for (AbstractLayer layer : neuralNet.getLayers()) {
            layer.setLearningRate(learningRate);
            layer.setMomentum(momentum);
            layer.setRegularization(regL2);
            layer.setBatchMode(batchMode);
            layer.setBatchSize(batchSize);
            layer.setOptimizerType(optType);
        }

        lossFunction = neuralNet.getLossFunction();

        float[] outputError;
        epoch = 0;
        totalTrainingLoss = 0;
        float prevTotalLoss = 0, totalLossChange;
        long startTraining, endTraining, trainingTime, startEpoch, endEpoch, epochTime;

        LOGGER.info("------------------------------------------------------------------------------------------------------------------------------------------------");
        LOGGER.info("TRAINING NEURAL NETWORK");
        LOGGER.info("------------------------------------------------------------------------------------------------------------------------------------------------");

        fireTrainingEvent(TrainingEvent.Type.STARTED);

        startTraining = System.currentTimeMillis();
        do {
            epoch++;
            lossFunction.reset();
            valLoss = 0;
            trainAccuracy = 0;
            valAccuracy = 0;

            if (shuffle) {
                trainingSet.shuffle();
            }
            int sampleCounter = 0;

            startEpoch = System.currentTimeMillis();

            for (MLDataItem dataSetItem : trainingSet) { // for all items in training set
                sampleCounter++;
                neuralNet.setInput(dataSetItem.getInput());
                outputError = lossFunction.addPatternError(neuralNet.getOutput(), dataSetItem.getTargetOutput().getValues());
                neuralNet.setOutputError(outputError);
                neuralNet.backward();

                if (!isBatchMode()) {
                    neuralNet.applyWeightChanges();
                } else if (sampleCounter % batchSize == 0) {
                    neuralNet.applyWeightChanges();

                    float miniBatchError = lossFunction.getTotal();
                    LOGGER.info("Epoch:" + epoch + ", Mini Batch:" + sampleCounter / batchSize + ", Batch Loss:" + miniBatchError);
                }
                fireTrainingEvent(TrainingEvent.Type.ITERATION_FINISHED); // BATCH_FINISHED?

                if (stopTraining) break; // if training was stoped externaly by calling stop() method
            }

            if (regL2 != 0) lossFunction.addRegularizationSum(regL2 * neuralNet.getL2Reg()); // 0.00001f

            endEpoch = System.currentTimeMillis();

            if (isBatchMode() && (trainingSamplesCount % batchSize != 0)) {
                neuralNet.applyWeightChanges();
            }

            totalTrainingLoss = lossFunction.getTotal();
            totalLossChange = totalTrainingLoss - prevTotalLoss;
            prevTotalLoss = totalTrainingLoss;
            trainAccuracy = calculateAccuracy(this.trainingSet);

            if (validationSet != null) {
                prevValLoss = valLoss;
                valLoss = validationLoss(validationSet);
                valAccuracy = calculateAccuracy(validationSet);
            }

            epochTime = endEpoch - startEpoch;

            if (validationSet != null)
                LOGGER.info("Epoch:" + epoch + ", Time:" + epochTime + "ms, TrainError:" + totalTrainingLoss + ", TrainErrorChange:" + totalLossChange + ", TrainAccuracy: " + trainAccuracy + ", ValError:" + valLoss + ", ValAccuracy: " + valAccuracy);
            else
                LOGGER.info("Epoch:" + epoch + ", Time:" + epochTime + "ms, TrainError:" + totalTrainingLoss + ", TrainErrorChange:" + totalLossChange + ", TrainAccuracy: " + trainAccuracy);


            if (Float.isNaN(totalTrainingLoss)) {
                stopTraining = true;
                LOGGER.info("The training was interrupted due to NaN value before completing all Epochs. Epochs completed: " + epoch + "/" + maxEpochs);
            }

            fireTrainingEvent(TrainingEvent.Type.EPOCH_FINISHED);

            // EARLY STOPPING
            if (earlyStopping && (epoch > 0 && epoch % checkpointEpochs == 0)) {
                if (prevCheckpointTestLoss - valLoss < earlyStoppingMinDelta) {
                    if (earlyStoppingCheckpointCount == earlyStoppingPatience) {
                        stop();
                    } else {
                        earlyStoppingCheckpointCount++;
                    }
                } else {
                    earlyStoppingCheckpointCount = 0;
                }

                // save network at this checkpoint since loss if going down
                prevCheckpointTestLoss = valLoss;
            }

            if (trainingSnapshots && (epoch > 0 && epoch % snapshotEpochs == 0)) {
                try {
                    FileIO.writeToFile(neuralNet, snapshotPath + "_epoch_" + epoch + ".dnet");
                } catch (IOException ex) {
                    LOGGER.catching(ex);
                }
            }

            stopTraining = stopTraining || ((epoch == maxEpochs) || (totalTrainingLoss <= maxError));

        } while (!stopTraining);

        endTraining = System.currentTimeMillis();
        trainingTime = endTraining - startTraining;

        LOGGER.info(System.lineSeparator() + "TRAINING COMPLETED");
        LOGGER.info("Total Training Time: " + trainingTime + "ms");
        LOGGER.info("------------------------------------------------------------------------");

        fireTrainingEvent(TrainingEvent.Type.STOPPED);
    }

    public long getMaxEpochs() {
        return maxEpochs;
    }

    public BackpropagationTrainer setMaxEpochs(long maxEpochs) {
        if (maxEpochs <= 0) {
            throw new IllegalArgumentException("Max epochs should be greater then zero : " + maxEpochs);
        }
        this.maxEpochs = maxEpochs;
        return this;
    }

    public float getMaxError() {
        return maxError;
    }

    public BackpropagationTrainer setMaxError(float maxError) {
        if (maxError < 0) {
            throw new IllegalArgumentException("Max error cannot be negative : " + maxError);
        }

        this.maxError = maxError;
        return this;
    }

    public BackpropagationTrainer setLearningRate(float learningRate) {
        if (learningRate < 0) {
            throw new IllegalArgumentException("Learning rate cannot be negative : " + learningRate);
        }
        if (learningRate > 1) {
            throw new IllegalArgumentException("Learning rate cannot be greater then 1 : " + learningRate);
        }

        this.learningRate = learningRate;

        return this;
    }

    public BackpropagationTrainer setL2Regularization(float regL2) {
        this.regL2 = regL2;
        return this;
    }

    public BackpropagationTrainer setL1Regularization(float regL1) {
        this.regL1 = regL1;
        return this;
    }

    public boolean getShuffle() {
        return shuffle;
    }

    public void setShuffle(boolean shuffle) {
        this.shuffle = shuffle;
    }

    private void fireTrainingEvent(TrainingEvent.Type type) {
        for (TrainingListener l : listeners) {
            l.handleEvent(new TrainingEvent(this, type));
        }
    }

    public void addListener(TrainingListener listener) {
        Objects.requireNonNull(listener, "Training listener cannot be null!");

        synchronized (listeners) {
            if (!listeners.contains(listener)) {
                listeners.add(listener);
            }
        }
    }

    public synchronized void removeListener(TrainingListener listener) {
        synchronized (listeners) {
            listeners.remove(listener);
        }
    }

    public boolean isBatchMode() {
        return batchMode;
    }

    public BackpropagationTrainer setBatchMode(boolean batchMode) {
        this.batchMode = batchMode;
        return this;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public BackpropagationTrainer setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public BackpropagationTrainer setMomentum(float momentum) {
        this.momentum = momentum;
        return this;
    }

    public float getMomentum() {
        return momentum;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void stop() {
        stopTraining = true;
    }

    public float getTrainingLoss() {
        return totalTrainingLoss;
    }

    public float getValidationLoss() {
        return valLoss;
    }

    public float getTrainingAccuracy() {
        return trainAccuracy;
    }

    public float getValidationAccuracy() {
        return valAccuracy;
    }


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

    public DataSet<?> getTestSet() {
        return validationSet;
    }

    public void setTestSet(DataSet<MLDataItem> testSet) {
        this.validationSet = testSet;
    }

    public boolean getEarlyStopping() {
        return earlyStopping;
    }

    public void setEarlyStopping(boolean earlyStopping) {
        this.earlyStopping = earlyStopping;
    }


    public BackpropagationTrainer setSnapshotPath(String snapshotPath) {
        this.snapshotPath = snapshotPath;
        return this;
    }

    public int getSnapshotEpochs() {
        return snapshotEpochs;
    }

    public void setSnapshotEpochs(int snapshotEpochs) {
        this.snapshotEpochs = snapshotEpochs;
    }

    public String getSnapshotPath() {
        return snapshotPath;
    }

    public boolean createsTrainingSnaphots() {
        return trainingSnapshots;
    }

    public void setTrainingSnapshots(boolean trainingSnapshots) {
        this.trainingSnapshots = trainingSnapshots;
    }

    public int getCheckpointEpochs() {
        return checkpointEpochs;
    }

    public BackpropagationTrainer setCheckpointEpochs(int checkpointEpochs) {
        this.checkpointEpochs = checkpointEpochs;
        return this;
    }

    public float getEarlyStoppingMinDelta() {
        return earlyStoppingMinDelta;
    }

    public BackpropagationTrainer setEarlyStoppingMinDelta(float earlyStoppingMinDelta) {
        this.earlyStoppingMinDelta = earlyStoppingMinDelta;
        return this;
    }

    public int getEarlyStoppingPatience() {
        return earlyStoppingPatience;
    }

    public BackpropagationTrainer setEarlyStoppingPatience(int earlyStoppingPatience) {
        this.earlyStoppingPatience = earlyStoppingPatience;
        return this;
    }


    /**
     * Sets properties from available keys in specified prop object.
     *
     * @param prop
     */
    public void setProperties(Properties prop) {
        this.maxError = Float.parseFloat(prop.getProperty(PROP_MAX_ERROR));
        this.maxEpochs = Integer.parseInt(prop.getProperty(PROP_MAX_EPOCHS));
        this.learningRate = Float.parseFloat(prop.getProperty(PROP_LEARNING_RATE));
        this.momentum = Float.parseFloat(prop.getProperty(PROP_MOMENTUM));
        this.batchMode = Boolean.parseBoolean(prop.getProperty(PROP_BATCH_MODE));
        this.batchSize = Integer.parseInt(prop.getProperty(PROP_BATCH_SIZE));
        this.optType = OptimizerType.valueOf(prop.getProperty(PROP_OPTIMIZER_TYPE));

        // iterate properties keys?use reflection to set them?
        //     this.maxLoss = Float.parseFloat(prop.getProperty(PROP_MAX_LOSS));
//        this.maxEpochs = Integer.parseInt(prop.getProperty(PROP_MAX_EPOCHS));

        if (prop.getProperty(PROP_LEARNING_RATE) != null)
            this.learningRate = Float.parseFloat(prop.getProperty(PROP_LEARNING_RATE));

//        this.momentum = Float.parseFloat(prop.getProperty(PROP_MOMENTUM));
//        this.batchMode = Boolean.parseBoolean(prop.getProperty(PROP_BATCH_MODE));
//        this.batchSize = Integer.parseInt(prop.getProperty(PROP_BATCH_SIZE));
//        this.optimizer = OptimizerType.valueOf(prop.getProperty(PROP_OPTIMIZER_TYPE));
    }

    // property names
    public static final String PROP_MAX_ERROR = "maxError";
    public static final String PROP_MAX_EPOCHS = "maxEpochs";
    public static final String PROP_LEARNING_RATE = "learningRate";
    public static final String PROP_MOMENTUM = "momentum";
    public static final String PROP_BATCH_MODE = "batchMode";
    public static final String PROP_BATCH_SIZE = "batchSize";      // for mini batch
    public static final String PROP_OPTIMIZER_TYPE = "optimizer";  // for mini batch


    private float validationLoss(DataSet<? extends MLDataItem> validationSet) {
        lossFunction.reset();
        float validationLoss = lossFunction.valueFor(neuralNet, validationSet);
        return validationLoss;
    }

    // only for classification problems
    private float calculateAccuracy(DataSet<? extends MLDataItem> validationSet) {
        EvaluationMetrics pm = eval.evaluate(neuralNet, validationSet);
        return pm.get(EvaluationMetrics.ACCURACY);
    }


    private void readObject(ObjectInputStream aInputStream) throws ClassNotFoundException, IOException {
        listeners = new ArrayList<>();
    }

}
