package deepnetts.util;

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;
import java.util.logging.Logger;

public class DeepNettsRunner {

    public static final String FILE_TRAINING_PROP = "training.properties";
    public static final String PROP_IMAGE_WIDTH = "imageWidth";
    public static final String PROP_IMAGE_HEIGHT = "imageHeight";
    public static final String PROP_LABELS_FILE = "labelsFile";
    public static final String PROP_TRAINING_FILE = "trainingFile";
    public static final String PROP_TEST_FILE = "testFile";
    public static final String PROP_LEARNING_RATE = "learningRate";
    public static final String PROP_MAX_ERROR = "maxError";

    private static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());

    public void run(String trainingPropFile) throws DeepNettsException, IOException {
//        LOGGER.set setLevel(Level.ALL);

        LOGGER.info("Starting training using " + trainingPropFile + " file...");

        // this should support .properties or json configuration
           // PropertiesConfiguration config = new  PropertiesConfiguration("training.properties");
            // load trainingSet properties and network architecture json, and then run that training
            Properties prop = new Properties();
            prop.load(new FileInputStream(("training.properties")));
            TypedProperties config = new TypedProperties(prop);

            int imageWidth = config.getInt(PROP_IMAGE_WIDTH);
            int imageHeight = config.getInt(PROP_IMAGE_HEIGHT);
            String labelsFile = config.getString(PROP_LABELS_FILE);
            String trainingFile = config.getString(PROP_TRAINING_FILE);
            float learningRate = config.getFloat(PROP_LEARNING_RATE);
            float maxError = config.getFloat(PROP_MAX_ERROR);

            // log all configuration settings
            // load network architecture configuration setting - json file
            // create network architecture from json

            // print out / log all settings from properties file to make it clear
            LOGGER.info("Loading images...");

            ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
            imageSet.loadLabels(new File(labelsFile));
            imageSet.loadImages(new File(trainingFile)); // napomena - putanje bi trebalo da budu relativne inace moraju da se regenerisu

            int labelsCount = imageSet.getLabelsCount();

            LOGGER.info("Done!");
            LOGGER.info("Creating neural network...");

            // u originanom LeCun-ovom radu koristi se tanh funkcija
            // create mnist architecture
            ConvolutionalNetwork convNet = ConvolutionalNetwork.builder()
                    .addInputLayer(imageWidth, imageHeight, 3)
                    .addConvolutionalLayer(6, 5)
                    .addMaxPoolingLayer(2, 2)
                    .addConvolutionalLayer(3, 3)
                    .addMaxPoolingLayer(2, 2)
                    .addFullyConnectedLayer(30)
                    .addFullyConnectedLayer(20)
                    .addOutputLayer(labelsCount, ActivationType.SOFTMAX) // softmax output // labelsCount
                    .lossFunction(LossType.CROSS_ENTROPY)
                    .randomSeed(123)
                    .build();
            LOGGER.info("Done!");
            LOGGER.info("Training neural network");

            // create a set of convolutional networks and do training, crossvalidation and performance evaluation
            BackpropagationTrainer trainer = new BackpropagationTrainer(convNet);
            trainer.setLearningRate(learningRate)
                    .setMaxError(maxError);
            //       .setMomentum(0.000)
            //     .setBatchMode(false);
            //.setBatchSize(10);
            trainer.train(imageSet);

            // how/where to get neural net from training is several nnets are theresult
//        ImageRecognizer imageRecognizer = new ImageRecognizer(convNet); // ConvolutionalImageRecognizer
//        ImageRecognitionResult results = imageRecognizer.recognize(image);
//        imageRecognizer.test(someTestSet);


    }

    public static void main(String[] args) throws IOException, DeepNettsException {
        String trainingPropFile;

        if (args.length == 0) {
            trainingPropFile = FILE_TRAINING_PROP;
        } else {
            trainingPropFile = args[0];
        }

        if (!(new File(trainingPropFile).exists())) {
            System.out.println("Cannot find training properties file: " + trainingPropFile);
            System.out.println("Make sure that file exists at specified path, or there is a training.properties file in running directory.");
        }

        (new DeepNettsRunner()).run(trainingPropFile);

        // ceo trening loguj i ispisi total training time
    }
}
