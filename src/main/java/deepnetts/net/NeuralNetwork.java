package deepnetts.net;

import deepnetts.accl.spi.AcceleratorService;
import deepnetts.accl.AcceleratorHandle;
import deepnetts.core.DeepNetts;
import deepnetts.eval.Evaluators;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.loss.BinaryCrossEntropyLoss;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.loss.LossFunction;
import deepnetts.net.loss.LossType;
import deepnetts.net.loss.MeanSquaredErrorLoss;
import deepnetts.net.train.Trainer;
import deepnetts.net.train.TrainerProvider;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.data.norm.AbstractScaler;
import java.io.IOException;
import java.io.ObjectInputStream;
import deepnetts.data.MLDataItem;
import deepnetts.data.Preprocessing;
import deepnetts.eval.ClassificationMetrics;
import deepnetts.eval.RegressionMetrics;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.tensor.Tensor;
import deepnetts.tensor.Tensor1D;
import deepnetts.util.DeepNettsThreadPool;
import deepnetts.util.FileIO;
import java.io.FileInputStream;
import java.lang.foreign.Arena;

/**
 * Base class for all neural networks in Deep Netts. Holds a list of abstract
 * layers and loss function. Provides methods for forward and backward
 * calculation, and to access input and output layers. Also provides network and
 * output labels.
 *
 * @see AbstractLayer
 * @see LossFunction
 *
 */
public class NeuralNetwork<T extends Trainer> implements TrainerProvider<T>, Serializable, AutoCloseable  {

    private static final long serialVersionUID = 1L;

    private T trainer;

    /**
     * Collection of all layers in this network (including input(first),
     * output(last) and hidden(in between)). As a minimum neural network must
     * have an input and output layer.
     */
    private List<AbstractLayer> layers;

    /**
     * Loss function Loss function represents total network error for some data,
     * and network learns by minimizing that error. Commonly used types of loss
     * functions are Mean Squared Error for regression problems and and Cross
     * Entropy for classification problems.
     */
    private LossFunction lossFunction;

    /**
     * Input layer. This layer accepts external inputs and sends them to the
     * next layer
     */
    private InputLayer inputLayer;

    /**
     * Output layer. This layer is the final step of processing network's input
     * and its output is network's output.
     */
    private OutputLayer outputLayer;

    /**
     * Labels for network outputs (classes)
     */
    private String[] outputLabels;

    private TensorBase inputWrapper; // is this used anywhere? should be removed but not to break serialization

    private Preprocessing<Tensor> preprocessing;

    private AbstractScaler normalizer;

    /**
     * Network's label
     */
    private String label;

    private float regularizationSum = 0;

    protected transient AcceleratorHandle cudaHandles;

    protected transient DeepNettsThreadPool threadPool;

    private Mode mode = Mode.INFERENCE;

    private transient Arena arena; // native memory for cuda
    
    protected NeuralNetwork() {
       // DeepNetts.checkLicense(); // if license is not valid this will throw exception
        layers = new ArrayList();
        if (DeepNetts.getInstance().useCuda()) {
            this.arena = Arena.ofConfined();

            //cudaHandles = new CudaHandles(arena);
            cudaHandles = AcceleratorService.defaultProvider().createAcceleratorHandle(arena); //a implementacija ce da instacira, uzmi default service provider i instacmiraj ga
        }

        // if multi threaded is on 
        if (DeepNetts.getInstance().isMultithreaded()) {
            threadPool = new DeepNettsThreadPool();
        }

    }

    /**
     * Sets network input and calculates entire network (triggers forward pass).
     *
     * @param inputs input tensor
     */
    public void setInput(TensorBase inputs) {
        // trebalo bi koristiti ceo preprocessing pipeline ne samo normalizer
        if (normalizer != null) {
            normalizer.scaleInput(inputs); // if there is a normalizer apply it - only for production not needed during training
        }
        inputLayer.setInput(inputs);
        forward();
    }

    @Deprecated
    public void setInput(float[] inputs) {
        setInput(new Tensor1D(inputs));
    }

    /**
     * Returns network's output.
     *
     * @return network's output
     */
    public float[] getOutput() {
        return outputLayer.getOutputs().getValues();
    }

    public TensorBase getOutputAsTensor() {
        return outputLayer.getOutputs();
    }

    /**
     * Sets the network's output errors, which are a difference between
     * actual(predicted) and target output.
     *
     * @param outputErrors array of errors for each output, a difference between
     * actual(predicted) and target output
     */
    public void setOutputError(TensorBase outputErrors) {
        outputLayer.setOutputErrors(outputErrors);
    }

    /**
     * Trains the neural network using specified training set.
     *
     * @param trainingSet example data given as (input, output) pairs to train
     * the network
     */
    public void train(DataSet<? extends MLDataItem> trainingSet) {
        trainer.train(trainingSet);
    }

    /**
     * Returns the prediction of this neural network for the given input. This
     * is the main method to use a trained neural network for
     * inference/prediction. A well trained neural network should provide
     * predictions with low error. Both input and returned prediction are
     * tensors, which are essentially multidimensional arrays.
     *
     * @param input input for the neural network
     * @return network's prediction.
     * @see TensorBase
     */
    public TensorBase predict(TensorBase input) {
        setInput(input);
        return getOutputAsTensor();
    }

    /**
     * Tests how good are predictions of this network using specified test set.
     * Automatically detects which type of task is network configured to perform
     * and applies appropriate evaluation/test procedure using corresponding
     * Evaluator.
     *
     * @param testSet data set to test/evaluate predictions
     * @return evaluation metrics that show how good this network is at
     * predicting unseen data
     *
     * @see EvaluationMetrics
     * @see ClassificationMetrics
     * @see RegressionMetrics
     * @see Evaluators
     */
    public EvaluationMetrics test(DataSet<? extends MLDataItem> testSet) {
        // check the loss and output function and use the appropriate evaluatior
        if (getLossFunction() instanceof CrossEntropyLoss
                || getLossFunction() instanceof BinaryCrossEntropyLoss) {
            return Evaluators.evaluateClassifier(this, testSet);
        } else {
            return Evaluators.evaluateRegressor(this, testSet);
        }
    }

    /**
     * Applies weight changes calculated in backward pass to all layers.
     */
    public void applyWeightChanges() {
        layers.forEach((layer) -> layer.applyWeightChanges()); // this can be parellelized since all layers are allraedy calculated - each layer cann apply changes in its own thread
    }

    /**
     * Performs a forward pass - calculations of all layers in the network for
     * the network's current input. This method iterates and calculates all
     * layers in this network. Assumes that network has previously set input
     * using <code>setInput</code> method.
     */
    protected void forward() {
        for (int i = 1; i < layers.size(); i++) {   // starts from 1 to skip input layer since it does not require any calculation
            layers.get(i).forward();
        }
    }

    /**
     * Performs a backward bass across all layers in neural network, which is
     * the calculation of corrections for the network internal parameters
     * (weights). This method invokes the training steps for all the layers
     * starting from the last/output layer and going backwards to first/input
     * layer.
     */
    public void backward() {
        // perfrom backward pass on all layers starting from last
        for (int i = layers.size() - 1; i > 0; i--) {
            layers.get(i).backward();
        }
    }

    /**
     * Adds a specified layer as a next layer in the network.
     *
     * @param layer layer to add to the network
     *
     * @see deepnetts.net.layers
     */
    protected void addLayer(AbstractLayer layer) {
        layers.add(layer);
    }

    /**
     * Gets layers of this neural network.
     *
     * @return layers of this neural network
     */
    public List<AbstractLayer> getLayers() {
        return layers;
    }

    public AbstractLayer getLayerAt(int idx) {
        return layers.get(idx);
    }

    /**
     * Returns the input layer of this neural network. Input layer is the first
     * layer in network which accepts the external input for the network, and
     * forwards it to the next layer in the network.
     *
     * @return input layer of this neural network
     *
     * @see InputLayer
     */
    public InputLayer getInputLayer() {
        return inputLayer;
    }

    /**
     * Returns the output layer of this network. Output layer is the last layer
     * of the network which provides final result of the network - predictions.
     *
     * @return output layer of this network.
     *
     * @see OutputLayer
     */
    public OutputLayer getOutputLayer() {
        return outputLayer;
    }

    /**
     * Sets output labels of this network.
     *
     * @param outputLabels labels which correspond to outputs of the network.
     */
    public void setOutputLabels(String... outputLabels) {
        this.outputLabels = outputLabels;
    }

    /**
     * Returns all labels for outputs of this network. Each output of the
     * network should have a label which describes what that output represents.
     *
     * @return labels for outputs of this network.
     */
    public String[] getOutputLabels() {
        return outputLabels;
    }

    /**
     * Gets a label of the i-th output this network. Each output of the network
     * should have a label which describes what that output represents.
     *
     * @param i idx position of the output
     * @return label for the i-th output
     */
    public String getOutputLabel(int i) {
        return outputLabels[i];
    }

    /**
     * Sets input layer of this network during the network building.
     *
     * @param inputLayer layer to be used as input of the network.
     *
     * @see InputLayer
     */
    protected void setInputLayer(InputLayer inputLayer) {
        this.inputLayer = inputLayer;
    }

    /**
     * Sets output layer of this network during the network building.
     *
     * @param outputLayer layer to be used as output.
     *
     * @see OutputLayer
     * @see SoftmaxOutputLayer
     */
    protected void setOutputLayer(OutputLayer outputLayer) {
        this.outputLayer = outputLayer;
    }

    /**
     * Returns a loss function of this network, which is used to calculate total
     * network error during the training.
     *
     * @return loss function of this network
     *
     * @see LossFunction
     * @see LossType
     */
    public LossFunction getLossFunction() {
        return lossFunction;
    }

    /**
     * Sets a loss function of this network, which is used to calculate total
     * network error during the training.
     *
     * @param lossFunction loss function to use during the training
     */
    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
        if (lossFunction instanceof MeanSquaredErrorLoss) {
            outputLayer.setLossType(LossType.MEAN_SQUARED_ERROR);
        } else if ((lossFunction instanceof CrossEntropyLoss) || (lossFunction instanceof BinaryCrossEntropyLoss)) {
            outputLayer.setLossType(LossType.CROSS_ENTROPY);
        }
    }

    /**
     * Returns the label(name) of this neural network
     *
     * @return label of this network
     */
    public String getLabel() {
        return label;
    }

    /**
     * Sets label(name) for this neural network.
     *
     * @param label label for this network
     */
    public void setLabel(String label) {
        this.label = label;
    }

    /**
     * Calculates and returns L2 regularization sum of the entire network (all
     * layers included). This value is used during the training to prevent
     * over-fitting.
     *
     * @return L2 regularization sum
     */
    public float getL2RegSum() {
        regularizationSum = 0;
        for (int i = 1; i < layers.size(); i++) {   // starts from 1 to skip input layer which dont have any weights
            regularizationSum += layers.get(i).getL2WeightSum();
        }
        return regularizationSum;
    }

    /**
     * Calculates and returns L1 regularization sum of the entire network (all
     * layers included). This value is used during the training to prevent
     * over-fitting.
     *
     * @return L2 regularization sum
     */
    public float getL1RegSum() {
        regularizationSum = 0;
        for (int i = 1; i < layers.size(); i++) {   // starts from 1 to skip input layer which dont have any weights
            regularizationSum += layers.get(i).getL1WeightSum();
        }
        return regularizationSum;
    }

    /**
     * Returns a training algorithm of this neural network. Training algorithm
     * performs tuning of the network's internal parameter(weights) in order to
     * minimize an error.
     *
     * @return training algorithm of this network
     *
     * @see BackpropagationTrainer
     */
    @Override
    public T getTrainer() {
        return trainer;
    }

    /**
     * Sets the training algorithm of this neural network.
     *
     * @param trainer training algorithm to use for this network
     *
     * @see BackpropagationTrainer
     */
    @Override
    public void setTrainer(T trainer) {
        this.trainer = trainer;
    }

    /**
     * Returns data normalization method that is applied to network's inputs.
     *
     * @return
     */
    public AbstractScaler getNormalizer() {
        return normalizer;
    }

    /**
     * Sets normalization data normalization method that is applied to network's
     * inputs.
     *
     * @param normalizer
     */
    public void setNormalizer(AbstractScaler normalizer) {
        this.normalizer = normalizer;
    }

    /**
     * Returns string representation of this network including all layers and
     * settings.
     *
     * @return string representation of this network
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        layers.stream().forEach(layer -> sb.append(layer.toString()).append(System.lineSeparator()));

        return sb.toString();
    }

    /**
     * Performs additional initialization after loading the network.
     *
     * @param ois
     * @throws ClassNotFoundException
     * @throws IOException
     */
    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
       // DeepNetts.checkLicense();
        ois.defaultReadObject();
        if (DeepNetts.getInstance().useCuda()) {            
          //  cudaHandles = new CudaHandles(arena);
            this.arena = Arena.ofConfined(); 
            cudaHandles = AcceleratorService.defaultProvider().createAcceleratorHandle(arena);
        }
        if (DeepNetts.getInstance().isMultithreaded()) {
            threadPool = new DeepNettsThreadPool();
        }

        layers.forEach((layer) -> {
            if (DeepNetts.getInstance().useCuda()) {
                layer.setCudaHandles(cudaHandles);
            }
            if (DeepNetts.getInstance().isMultithreaded()) {
                layer.setThreadPool(threadPool);
            }

            layer.initTransientFields(); // ovo sam radio zbog threadova,ali ne i smeo jer zeznamen deserijalizovane tezine. posebna metoda za runtime         
        });
    }

    /**
     * Gets preprocessing that needs to be performed before input is fed to this
     * network.
     *
     * @return
     */
    public Preprocessing<Tensor> getPreprocessing() {
        return preprocessing;
    }

    /**
     * Sets preprocessing that needs to be performed before input is fed to this
     * network.
     *
     * @param preprocessing
     */
    public void setPreprocessing(Preprocessing<Tensor> preprocessing) {
        this.preprocessing = preprocessing;
    }

    /**
     * Saves this network using serialization to file with specified fileName.
     *
     * @param fileName name of the file to save network
     * @throws IOException
     */
    public void save(String fileName) throws IOException {
        FileIO.writeToFile(this, fileName);
    }

    /**
     * Loads and returns neural network previously saved to a file.
     *
     * @param <T> type(class) of the network to lead and return.
     * @param fileName name of the file to load network from
     * @param clazz class of the neural network to load
     * @return loaded neural network
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public static <T> T load(String fileName, Class<T> clazz) throws IOException, ClassNotFoundException {
        T neuralNet;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fileName))) {
            neuralNet = clazz.cast(ois.readObject());
        }
        return neuralNet;
    }

    public Mode getMode() {
        return mode;
    }

    public void setMode(Mode mode) {
        this.mode = mode;
        for (AbstractLayer layer : getLayers()) {
            layer.setMode(mode);
        }
    }

    public DeepNettsThreadPool getThreadPool() {
        return threadPool;
    }

    @Override
    public void close() throws Exception {
        arena.close(); // ovde bi zapravi trebao i da prodjes kroz graph svih cuda tenzora. da oslobodis gpu i native memoriju, treba mi graph za to, i da prodjem kroz sve nodes/tenzore i druge pomocne memorije koje se kreiraju u rADU
    }

}
