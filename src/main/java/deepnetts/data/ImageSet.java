/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.
 */
package deepnetts.data;

import deepnetts.core.DeepNetts;
import deepnetts.tensor.Tensor1D;
import deepnetts.util.DeepNettsException;
import deepnetts.util.DeepNettsThreadPool;
import deepnetts.util.ImageResize;
import deepnetts.util.ImageSetUtils;
import static deepnetts.util.ImageSetUtils.IMAGE_IDX_FILE;
import static deepnetts.util.ImageSetUtils.LABELS_FILE;
import deepnetts.util.ImageUtils;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.TensorBase;
import deepnetts.util.Stats;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.visrec.ml.data.DataSet;

/**
 * Data set with images that will be used to train convolutional neural network.
 */
public class ImageSet extends TabularDataSet<ExampleImage> {

    private final int imageWidth;
    private final int imageHeight;
    private boolean scaleImages = true;
    private boolean invertImages = false;
    private boolean cropCornersAndCenter=false;
    private TensorBase mean;
    private String delimiter = " "; // delimiter between image and label 
    private ImageResize resizeStrategy = ImageResize.STRATCH;
    private static String NEGATIVE_LABEL = "negative";
    private static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());

    // mozda dodaj Builder sa svim opcijama
    
    /**
     * Indicates if zero mean has been performed on this data set
     */
    private boolean zeroMeanPixels;
    
    /**
     * Augmentation options
     */
    private boolean flipHorizontal, brightness, grayscale, translate;

    /**
     * Creates empty image set for images of specified size.
     * 
     * @param imageWidth
     * @param imageHeight 
     */
    public ImageSet(int imageWidth, int imageHeight) {
        super();
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
    }

    /**
     * Creates image set with images from specified directory path.
     * 
     * @param imageWidth all images in set will be scaled to this width
     * @param imageHeight  all images in set will be scaled to this height
     * @param imageDirPath path to the root directory which contains subdirectories with images 
     * @throws java.io.IOException 
     */    
    public ImageSet(int imageWidth, int imageHeight, String imageDirPath) throws IOException {
        super();
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        setScaleImages(true);
        
        // da li vec postoji labels.txt fajl? ako nema kreiraj ga
        if (!Files.exists(Paths.get(imageDirPath+"/"+LABELS_FILE)) ) {
            ImageSetUtils.createLabelsIndex(imageDirPath);
        }                   
        
        // da li vec postoji index.txt ili images.txt fajl? ako nema kreiraj ga
        if (!Files.exists(Paths.get(imageDirPath+"/"+IMAGE_IDX_FILE)) ) {
            ImageSetUtils.createImageIndex(imageDirPath);
        }
         
        loadLabels(imageDirPath+"/"+LABELS_FILE);
        loadImages(imageDirPath+"/"+IMAGE_IDX_FILE);        
    }

    final private Object LOCK = new Object();

    /**
     * Adds an example image that will be used to train deep learning model to this set.
     * 
     * @param exImage holds raw image pixel data and corresponding image label
     * @return this instance of ImageSet
     * @throws DeepNettsException if image is empty or has wrong dimensions.
     */
    @Override
    public DataSet<ExampleImage> add(ExampleImage exImage) throws DeepNettsException {
        if (exImage == null) {
            throw new DeepNettsException("Example image cannot be null!");
        }
        synchronized (LOCK) {
            if ((exImage.getWidth() == imageWidth) && (exImage.getHeight() == imageHeight)) {
                items.add(exImage);
            }
        }

        return this;
    }

    /**
     * Loads images from the specified image index file.
     * @param imageIdxFile txt file with list of images
     * 
     * @throws FileNotFoundException 
     */
    public void loadImages(String imageIdxFile) throws FileNotFoundException {
        loadImages(new File(imageIdxFile));
    }

    /**
     * Loads example images with corresponding labels from the specified file.
     *
     * @param imageIdxFile Plain text file that contains space delimited image
     * paths and labels
     * @throws java.io.FileNotFoundException if imageIdxFile was not found
     */
    public void loadImages(File imageIdxFile) throws FileNotFoundException {
        // TODO: First load entire image index, then load and preprocess image in
        // multithreaded way TODO2: load images in batches  verovtano neki iterator nextBatch()
        // ako index fajl ne postoji kreiraj ga pomocu utility metode!
        // isto i labels file da se minimizuje znanje o low level detaljima posebno trivijalnim
        
        Objects.requireNonNull(imageIdxFile, "Index file cannot be null!");
        if (columnNames == null) {
            throw new DeepNettsException("Error: Labels are not loaded. In order to load images correctly you have to load labels first using ImageSet.loadLabels method.");
        }

        // use paths of the image index file as root path for image categories
        final String rootPath = imageIdxFile.getPath().substring(0, imageIdxFile.getPath().lastIndexOf(File.separator));

        String imgFileName = null;
        String label = null;

        List<BufferedImage> imagesBuffer = new LinkedList<>();
        List<String> labelsBuffer = new LinkedList<>();
        List<String> imageFilesBuffer = new LinkedList<>();

        // TODO: da radi u batch-u. Da ima interni brojac dokle je stigao. Ili da drzi otvoren stream da iam metodu loadNextBatch() mozda to najbolje u posebnoj metodi ako je mod za trening batch.
        // TODO: napravi ovo asinhrono da ucitava i preprocesira u posebnim threadovima, u perspektivi u batchovima, ne sve odjendnom
        // ucitaj prvo indeks slika a onda ucitavanje i preprocrsiranje slika parelelizuj da jedan thread radi ucitavanje a drugi preprocsiranje onoga sto je ucitano
        try ( BufferedReader br = new BufferedReader(new FileReader(imageIdxFile))) {
            String line = null;
            int lineCount = 0;

            // we can also catch and log FileNotFoundException, IOException in this loop
            while ((line = br.readLine()) != null) {
                lineCount++;
                if (line.isEmpty()) { // skip empty lines
                    continue;
                }
                String[] parts = line.split(delimiter); // parse file and class label from current line - sta ako naziv fajla sadrzi space? - to ne sme ili detektuj nekako sa lastIndex
                if (parts.length > 2) {
                    throw new DeepNettsException("Bad file format: image paths and labels should not contain spaces! At line " + lineCount);
                }

                imgFileName = parts[0];

                if (parts.length == 2) { // use specified label if it is available
                    label = parts[1];
                } else if (parts.length == 1) {  // otherwise use name of parent folder as a label
                    final int labelEndIdx = imgFileName.lastIndexOf(File.separator); // assumes one top directory which corresponds to category label
                    label = imgFileName.substring(0, labelEndIdx);
                }
                //  String shortFileName = parts[0].substring(parts[0].indexOf(File.separator)+1);
                imgFileName = rootPath + File.separator + imgFileName;

                BufferedImage image = ImageIO.read(new File(imgFileName));
                if (image == null) continue; // if not image file skip
                imagesBuffer.add(image);
                imageFilesBuffer.add(imgFileName);
                labelsBuffer.add(label);
                LOGGER.info(imgFileName); // log loaded images for debugging
                
                // ubaci ovo i u drugu metodu za load images - batchSize ili loading buffer size d amoze da se setuje metodom ko hoce ali ne mora, pa ako bas neko ima problem...
//                if (lineCount % 3000 == 0) {    // batchSize umesto 1000 ovde
//                    processImages(imagesBuffer, labelsBuffer);                    
//                }
                
            }

            processImages(imageFilesBuffer, imagesBuffer, labelsBuffer); // ovo da radi dok ucitava slike, ili kako koju procesira tako da je izbaci iz liste
            imagesBuffer.clear();
            imageFilesBuffer.clear();
            labelsBuffer.clear();  
            
            
            if (isEmpty()) {
                throw new DeepNettsException("Zero images loaded!");
            }

            LOGGER.info("Loaded " + size() + " images");

        } catch (FileNotFoundException ex) {
            LOGGER.info("File Could not find image file"+ex.getMessage()); // Lychee : 489
            throw new DeepNettsException("Could not find image file: " + imgFileName, ex);
        } catch (IOException ex) {
            LOGGER.info("Error loading image file: " + ex.getMessage());
            throw new DeepNettsException("Error loading image file: " + imgFileName, ex);
        } catch (NullPointerException ex) {
            LOGGER.info(ex.getMessage());
            throw new DeepNettsException("Error loading image file: " + imgFileName, ex);
        }
    }

    /**
     * Loads specified number of example images with corresponding labels from
     * the specified file.
     *
     * @param imageIdxFile Plain text file which contains space delimited image
     * file paths and label
     * @param numOfImages number of images to load
     */
    public void loadImages(File imageIdxFile, int numOfImages) throws DeepNettsException {
        Objects.requireNonNull(imageIdxFile, "Index file cannot be null!");

        if (columnNames == null) {
            throw new DeepNettsException("Error: Labels are not loaded. In order to load images correctly you have to load labels first using ImageSet.loadLabels method.");
        }

        final String rootPath = imageIdxFile.getPath().substring(0, imageIdxFile.getPath().lastIndexOf(File.separator));

        String imgFileName = null;
        String label = null;

        List<BufferedImage> imagesBuffer = new LinkedList<>();
        List<String> imageFilesBuffer = new LinkedList<>();
        List<String> labelsBuffer = new LinkedList<>();

        // ako je numOfImages manji od broja slika u fajlu logovati
        try ( BufferedReader br = new BufferedReader(new FileReader(imageIdxFile))) {
            String line = null;

            for (int i = 0; i < numOfImages; i++) {
                line = br.readLine();
                if (line == null || line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split(delimiter); // parse file and class label from line

                if (parts.length > 2) {
                    throw new DeepNettsException("Bad file format: image paths and labels should not contain spaces! At line " + i);
                }

                imgFileName = parts[0];

                if (parts.length == 2) { // use specified label if it is specified
                    label = parts[1];
                } else if (parts.length == 1) {  // otherwise use the name of a parent folder as label
                    final int labelEndIdx = imgFileName.lastIndexOf(File.separator); // assumes one top directory which corresponds to category label
                    label = imgFileName.substring(0, labelEndIdx);
                }

                imgFileName = rootPath + File.separator + imgFileName;
                final BufferedImage image = ImageIO.read(new File(imgFileName)); // na pozicije 20 i 21 ubaci null slike
                // ucitavanje U xample image prosledi mu fajl da pamti i fajl??? zauzima memoriju nepotrebno, ali najbolje za testiranje jer resava preprocesiranje
                // dodaj konstruktor sa putanjom???
                if (image == null) continue; // if not image file skip                
                imagesBuffer.add(image);
                imageFilesBuffer.add(imgFileName);                
                labelsBuffer.add(label);
                LOGGER.info(imgFileName);               
            }
        } catch (FileNotFoundException ex) {
            LOGGER.info(ex.getMessage());
            throw new DeepNettsException("Could not find image file: " + imgFileName, ex);
        } catch (IOException ex) {
            LOGGER.info(ex.getMessage());
            throw new DeepNettsException("Error loading image file: " + imgFileName, ex);
        }

        processImages(imageFilesBuffer, imagesBuffer, labelsBuffer);
        imagesBuffer.clear();
        imageFilesBuffer.clear();
        labelsBuffer.clear();           

        // sacekaj da pool zavrsi
        if (isEmpty()) {
            throw new DeepNettsException("Zero images loaded!");
        }
        LOGGER.info("Loaded " + size() + " images");
    }

    private void processImages(List<String> imageFiles, List<BufferedImage> images, List<String> labels) {
        int threadCount = DeepNettsThreadPool.getMaxThreadsNum();
        List<Callable<Boolean>> workers = new ArrayList<>();
        int imagesPerWorker = images.size() / threadCount;
        int start = 0, end = 0;
        CountDownLatch latch = new CountDownLatch(threadCount);
        for (int t = 0; t < threadCount; t++) {
            end = start + imagesPerWorker;
            if (end > images.size()) { //????? ovo je ruzni hack da ovo ipak radiu jednom threadu!!!!
                end = images.size();
            } else if ((t == threadCount-1) && (end < images.size() )) {
                end = images.size();                
            }

            ImageProcessor imgProc = new ImageProcessor(imageFiles, images, labels, start, end, latch);
            workers.add(imgProc);
            start = end;
        }     
        ExecutorService es = Executors.newFixedThreadPool(threadCount);
        try {
            List<Future<Boolean>> results = es.invokeAll(workers);
            //latch.await();// ovo nam ne treba sa invoke all on bi trebalo da ceka... ali kao zbog shutdowna, medjutim ne ceka,,, mora await
            es.shutdown();
        } catch (InterruptedException ex) {
            java.util.logging.Logger.getLogger(ImageSet.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        if (zeroMeanPixels) {
            zeroMean();
        }
        
    }

    private class ImageProcessor implements Callable<Boolean> {

        private final List<BufferedImage> images; // najbolje bi bilo da ne ubacuju po jedan jer ce se dosta cekati nego  ceo batch koji uzme svaki thread kad zavrsi svoj deo, dakle da imaju svoju kolekciju u koju ce da ubacuju a ne da svi ubacuju u jednu
        // neki fazon  ThreadLocal images vidihttps://www.baeldung.com/java-threadlocal
        private final List<String> labels;
        private final int start;
        private final int end;
        private final CountDownLatch latch;
        private final List<String> imageFiles;


        public ImageProcessor(List<String> imageFiles, List<BufferedImage> images, List<String> labels, int start, int end, CountDownLatch latch) {
            this.images = images;
            this.imageFiles=imageFiles;
            this.labels = labels;
            this.start = start;
            this.end = end;
            this.latch = latch;
        }
        
        @Override
        public Boolean call() throws IOException {
            for (int i = start; i < end; i++) {
                BufferedImage img = images.get(i);
                String imgFile = imageFiles.get(i);
                final String label = labels.get(i); // ovde bolje
                
                images.set(i, null);
                imageFiles.set(i, null);
                labels.set(i, null);
                
                if (scaleImages) {
                    switch(resizeStrategy) {
                        case STRATCH:   
                            img = ImageUtils.scaleImage(img, imageWidth, imageHeight);
                        break;    
                        case CENTER:   
                            img = ImageUtils.scaleAndCenter(img, imageWidth, imageHeight, 0, Color.WHITE);
                        break;      
                        default:
                            throw new RuntimeException("Unknown image scaling method: "+resizeStrategy);
                    }
                }

                ExampleImage exImg = null;
                if (grayscale) {
                    img = ImageUtils.grayscale(img);
                    exImg = new ExampleImage(img, label, 1);
                } else {
                    exImg = new ExampleImage(new File(imgFile), img, label); 
                }
                
                //ExampleImage exImg = new ExampleImage(new File(imgFile), img, label);  // kako da kazem ovima da cuva buffered img kad en moze ako ih ima puno???
                Tensor1D targetOutput = new Tensor1D(oneHotEncode(label, columnNames));
                
                exImg.setTargetOutput(targetOutput);
                add(exImg);

                // treba ga invertovati i ako je grayscale!
                if (invertImages) {
                    exImg.invert();
                }
                                
                if (cropCornersAndCenter) { // zoom and random crop
                    // zoom or downsize to 25% larger image than target size, samo ga nije invertovao!!!!
                    img = ImageUtils.scaleBySmallerTarget(img, (int)(imageWidth*1.25), (int)(imageHeight*1.25));
                   
                    List<BufferedImage> cropedImages = ImageUtils.cropAtCornersAndCenter(img, imageWidth, imageHeight); // ne moze ivde da idu dimenzije slike ne
                    cropedImages.forEach((crpImg) -> {
                        ExampleImage exCrpImg = new ExampleImage(crpImg, label);
                        exCrpImg.setTargetOutput(targetOutput);
                        add(exCrpImg);
                    });
                }        
                
                // ovo ovde je za augmentaciju slike vec dodate iznad
                // treba jedna grana za augmentaciju, a jedna za preprocesiranje ucitanih , isprojektuj logiku if-ova i sta gde ide
                // preprocesiranje: skaliranje, invert, grayscale
                // augmentacija: flip, translate, grayscale, brightness
                
                try {
                    if (flipHorizontal) {
                        BufferedImage flpdImg = ImageUtils.flipHorizontal(img);
                        ExampleImage teximg = new ExampleImage(flpdImg, exImg.getLabel());
                        teximg.setTargetOutput(targetOutput);
                        add(teximg); // add fliped image
                    }

                    // ovaj dodaje grayscale kao augmentaciju! kao dodatnu sliku
//                    if (grayscale) {
//                        BufferedImage grayImg = ImageUtils.grayscale(img);
//                        ExampleImage teximg = new ExampleImage(grayImg, exImg.getLabel(), 1); // use only one channel for grayscale images
//                        if (invertImages) teximg.invert();
//                        teximg.setTargetOutput(targetOutput);
//                        add(teximg);
//                    }

                    if (brightness) {
                        BufferedImage rbrImg = ImageUtils.randomTintAndBrightness(img); // izgenerisi nekoliko varijanti ovde kao za translate
                        ExampleImage teximg = new ExampleImage(rbrImg, exImg.getLabel());
                        teximg.setTargetOutput(targetOutput);
                        add(teximg);
                    }

                    if (translate) {
                        List<BufferedImage> translated = ImageUtils.translateImage(img);
                        final String tlabel = exImg.getLabel();
                        translated.forEach((tImg) -> {
                            ExampleImage teximg = new ExampleImage(tImg, tlabel);
                            teximg.setTargetOutput(targetOutput);
                            add(teximg);
                        });
                    }

                } catch (DeepNettsException ex) {
                    throw new RuntimeException(ex.getMessage(), ex);
                }

            }
          //  latch.countDown();
            
            return Boolean.TRUE;
        }
    }

    /**
     * Creates and returns binary array for specified label using
     * one-hot-encoding scheme. Each position in array corresponds to one label,
     * position with label given as parameter is 1, while other positions are
     * zero. Returns all zeros for label 'negative'.
     *
     * TODO: maybe to greate map and just get corresponding vector for each
     * 
     * @param label specific tabel to encode with 1 in return vector
     * @param labels all available labels
     * @return
     */
    private float[] oneHotEncode(final String label, final String[] labels) { // todo: replace with actutal vector size
        final float[] returnArr = new float[labels.length];

        if (label.equalsIgnoreCase(NEGATIVE_LABEL)) {
            return returnArr;
        }
        for (int i = 0; i < labels.length; i++) {
            if (labels[i].equals(label)) {
                returnArr[i] = 1;
            }
        }

        return returnArr;
    }

    public int getLabelsCount() {
        return columnNames.length;
    }

    /**
     * Splits data set into several parts specified by the input parameter
     * partSizes. Values of partSizes parameter represent the sizes of data set
     * parts that will be returned. Part sizes are integer values that represent
     * percents, cannot be negative or zero, and their sum must be 100
     *
     * @param partSizes sizes of the parts in percents
     * @return parts of the data set of specified size
     */
    @Override
    public ImageSet[] split(double... partSizes) {
        if (partSizes.length < 2) {
            throw new IllegalArgumentException("Must specify at least two parts");
        }
        int partsSum = 0;
        for (int i = 0; i < partSizes.length; i++) {
            if (partSizes[i] <= 0) {
                throw new IllegalArgumentException("Value of the part cannot be zero or negative!");
            }
            partsSum += partSizes[i];
        }

        if (partsSum > 1) {
            throw new IllegalArgumentException("Sum of parts/percents cannot be larger than 1!");
        }

        LOGGER.info("Splitting data set: " + Arrays.toString(partSizes));

        ImageSet[] subSets = new ImageSet[partSizes.length];
        int itemIdx = 0;

        this.shuffle();

        for (int p = 0; p < partSizes.length; p++) {
            ImageSet subSet = new ImageSet(imageWidth, imageHeight);
            int itemsCount = (int) (size() * partSizes[p]);
            subSet.setInvertImages(invertImages);
            subSet.zeroMeanPixels = zeroMeanPixels;  
            subSet.mean = mean;

            for (int j = 0; j < itemsCount; j++) {
                subSet.add(items.get(itemIdx));
                itemIdx++;
            }

            subSets[p] = subSet;
            subSets[p].columnNames = columnNames;
            // anything else? image dimensions?
        }

        return subSets;
    }

    /**
     * Loads and returns image labels to train neural network from the specified
     * file. These labels will be used to label network's outputs.
     *
     * @param filePath
     * @return
     * @throws DeepNettsException
     */
    public String[] loadLabels(String filePath) throws DeepNettsException {
        return loadLabels(new File(filePath));
    }

    /**
     * Loads and returns image labels to train neural network from the specified
     * file.These labels will be used to label network's outputs.
     *
     * @param file file to load labels from
     * @return
     * @throws DeepNettsException
     */
    public String[] loadLabels(File file) throws DeepNettsException {
        try ( BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line = null;
            List<String> labelsList = new ArrayList<>(); // temporary labels list
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) continue; // skip empty lines
                
                line = line.trim();
                if (line.contains(" ")) {
                    throw new DeepNettsException("Bad label format: Labels should not contain space characters! For label:" + line);
                }
                labelsList.add(line);
            }
            this.columnNames = labelsList.toArray(new String[labelsList.size()]);
            LOGGER.info("Loaded " + labelsList.size() + " labels");
            return columnNames;
        } catch (FileNotFoundException ex) {
            LOGGER.info("Could not find labels file: " + file.getAbsolutePath());
            throw new DeepNettsException("Could not find labels file: " + file.getAbsolutePath(), ex);
        } catch (IOException ex) {
            LOGGER.info("Error reading labels file: " + file.getAbsolutePath());
            throw new DeepNettsException("Error reading labels file: " + file.getAbsolutePath(), ex);
        }
    }

    /**
     * Applies zero mean normalization to entire dataset, and returns mean
     * tensor.
     *
     * @return mean Tensor for the entire dataset
     */
    public TensorBase zeroMean() {
        zeroMeanPixels = true;
        
        mean = new Tensor3D(3, imageHeight, imageWidth); // fix: sta ako je 1d tenzor grayscale? onda ovo iznad ali pazi grayscale je prvobitno zamisljen za augmentaciju

        // napravi i median!!!
        // prodji kroz ceo data set i za zvaki pixel nadlji
        
        // sum all matrices - mean po kanalu, i onda od njih mean za ceo data set
        items.forEach((img) ->  { mean.add(img.getInput()); } ); // ovde treba da ispadne jedan tenzor sa mean po kanalu: znaci mean od mean od svakog kanala 
 
        // divide by number of images
        mean.div(items.size());
                
        // subtract mean from each image
        for (ExampleImage image : items) {
            image.getInput().sub(mean);
        }

        return mean;
    }
    
    public TensorBase zeroMeanPerChannel() {
        zeroMeanPixels = true;
        
        mean = new Tensor3D(3, imageHeight, imageWidth);
        float[][] imgChMean = new float[items.size()][3];
        
        // izracunaj i std po kanalu za sliku i std za ceo data set, za normalizaciju, kako navuci na pozitivi ceo
        
        // mozda bolje da sve saberem to je globalni mean - sve piksele iz svih kanala
        // ili sve piskele iz pojedinacnih kanala i ond aza ceo data set
        int i =0;
        for(ExampleImage img : items) {
            final Tensor3D input = img.getInput();
            for(int ch=0; ch<3; ch++) {
                imgChMean[i][ch] = input.channelMean(ch);
            }
            i++;
        }
        
        
        float[] dsMean = new float[3];
        for(int j=0; j<items.size(); j++) {
            dsMean[0] += imgChMean[j][0];
            dsMean[1] += imgChMean[j][1];
            dsMean[2] += imgChMean[j][2];
        }
        
        dsMean[0] = dsMean[0] / items.size();
        dsMean[1] = dsMean[1] / items.size();
        dsMean[2] = dsMean[2] / items.size();

        mean = new Tensor3D(3, 1, 1, dsMean);

        // subtract mean from each image 
        for (ExampleImage img : items) {
            Tensor3D inTensor = img.getInput(); // ne moze  ovako jer mora istu vrednost od svih da oduzme - morao bi broad casting ali hajede prvo rucno
            // dodaj svim aistu vrednost add
            // podeli sa 2
            for (int ch = 0; ch < 3; ch++) {
                for (int c = 0; c < inTensor.cols(); c++) {
                    for (int r = 0; r < inTensor.rows(); r++) {
                        inTensor.sub(dsMean[ch], ch, r, c);
                    //    inTensor.add(1, ch, r, c); // move to positive
                    }
                }
            }
            //inTensor.div(2); // move to positive
        }
        
        // podeli sa std po kanalu
        
        
        return mean;
    }    
    

// racunaj total mean and total std - izracunaj po skracenoj formuli za sve kanale
    // https://kozodoi.me/blog/20210308/compute-image-stats
    // https://www.thoughtco.com/sum-of-squares-formula-shortcut-3126266
    public TensorBase zeroMeanAndNormalize() {
        zeroMeanPixels = true;
                        
        Stats[] chStats = new Stats[3]; // stats for each channel
        chStats[0] = new Stats();
        chStats[1] = new Stats();
        chStats[2] = new Stats();
        
        // izracunaj i std po kanalu za sliku i std za ceo data set, za normalizaciju, kako navuci na pozitivi ceo
        
        float[] sum = new float[3], sqrSum = new float[3];
        float n = imageWidth * imageHeight * items.size();
        for(ExampleImage img : items) {
            final Tensor3D input = img.getInput();
            for(int ch=0; ch<3; ch++) {
                for(int c=0; c<input.cols(); c++) {
                    for(int r=0; r<input.rows(); r++) {
                        final float val = input.get(ch, r, c);
                        sum[ch] += val;
                        sqrSum[ch] += val*val;
                    }
                }
            }
        }

        for(int ch=0; ch<3; ch++) {
            float chMean = sum[ch] / n;
            float std = (float)Math.sqrt(sqrSum[ch] - (sum[ch]*sum[ch])/n);                
                
            chStats[ch].setMean(chMean);
            chStats[ch].setStd(std);  // kako std za ceo data set - isto ako i za mean sve pixele pa sta bude      
        }
                               
        mean = new Tensor3D(3, 1, 1);

        // subtract mean from each image 
        for (ExampleImage img : items) {
            Tensor3D inTensor = img.getInput(); // ne moze  ovako jer mora istu vrednost od svih da oduzme - morao bi broad casting ali hajede prvo rucno
            // dodaj svim aistu vrednost add
            // podeli sa 2
            for (int ch = 0; ch < 3; ch++) {
                final float chMean = chStats[ch].getMean();
                for (int c = 0; c < inTensor.cols(); c++) {
                    for (int r = 0; r < inTensor.rows(); r++) {
                        inTensor.sub(chMean, ch, r, c);
                        inTensor.divChannel(chStats[ch].getStd(), ch); // divide each channel with its std .. ovo bolje sa nekom funkcijom koja rad po kanalima
                    }
                }
            }
        }
        
        return mean; // fix, and fiil this one
    }        
    
    /**
    public Tensor zeroMedian() {
        boolean zeroMedianPixels = true;
        // kako izracunati medijanu na svim pixelima?
        Tensor median = new Tensor(imageHeight, imageWidth, 3);
        * sortiraj niz tj kopiju nizad i uzmi el iz sredine
        // kako sortirati niz vectora po pozicijama
        // a da ne remetim originalne vektore, treba mi duplikat celog data seta da ga uradim
        // uzimam jedan pojedan ubacujem gde treba i ostale shftujem/pomeram
                
        // subtract mean from each image
        for (ExampleImage image : items) {
            image.getInput().sub(median);
        }

        return median;
    }    
    */

    /**
     * Returns flag that indicates wheather images should be scaled to specified
     * dimensions while creating image set.
     *
     * @return
     */
    public boolean getScaleImages() {
        return scaleImages;
    }

    public final void setScaleImages(boolean scaleImages) {
        this.scaleImages = scaleImages;
    }

    public boolean getInvertImages() {
        return invertImages;
    }

    public ImageSet setInvertImages(boolean invertImages) {
        this.invertImages = invertImages;
        return this;
    }

    public boolean getFlipHorizontal() {
        return flipHorizontal;
    }

    public boolean getBrightness() {
        return brightness;
    }

    public boolean getGrayscale() {
        return grayscale;
    }

    public void setFlipHorizontal(boolean flipHorizontal) {
        this.flipHorizontal = flipHorizontal;
    }

    public void setBrightness(boolean brightness) {
        this.brightness = brightness;
    }

    public void setGrayscale(boolean grayscale) {
        this.grayscale = grayscale;
    }

    public boolean getTranslate() {
        return translate;
    }

    public void setTranslate(boolean translate) {
        this.translate = translate;
    }    

    public boolean getCropCornersAndCenter() {
        return cropCornersAndCenter;
    }

    public void setCropCornersAndCenter(boolean cropCornersAndCenter) {
        this.cropCornersAndCenter = cropCornersAndCenter;
    }
    
    
    
    /**
     * Returns output/image labels.
     *
     * @return
     */
    @Override
    public String[] getTargetColumnsNames() {
        return columnNames;
    }

    public Map<String, Integer> countByClasses() {
        HashMap<String, Integer> map = new HashMap<>();

        for (ExampleImage item : items) {
            if (map.containsKey(item.getLabel())) {
                final String key = item.getLabel();
                map.put(key, map.get(key) + 1);
            } else {
                map.put(item.getLabel(), 1);
            }
        }

        LOGGER.info("Number of images by label/class");
        for (String key : map.keySet()) {
            LOGGER.info(key + " : " + map.get(key));
        }

        return map;
    }

    public String getDelimiter() {
        return delimiter;
    }

    public void setDelimiter(String delimiter) {
        this.delimiter = delimiter;
    }

    public ImageResize getResizeStrategy() {
        return resizeStrategy;
    }

    public ImageSet setResizeStrategy(ImageResize resizeStrategy) {
        this.resizeStrategy = resizeStrategy;
        return this;
    }

    public int getImageWidth() {
        return imageWidth;
    }

    public int getImageHeight() {
        return imageHeight;
    }

    public TensorBase getMean() {
        return mean;
    }

    public boolean getZeroMeanPixels() {
        return zeroMeanPixels;
    }
    
//    public ImageSet setZeroMeanPixels(boolean zeroMean) {
//        this.zeroMeanPixels = zeroMean;
//        return this;
//    }

}
