/**  
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation 
 *  based learning and image recognition.
 * 
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */
    
package deepnetts.util;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;

/**
 * Utility methods to work with images.
 * 
 * 
 * https://machinelearningmastery.com/best-practices-for-preparing-and-augmenting-image-data-for-convolutional-neural-networks/
 * random rescaling, horizontal flips, perturbations to brightness, contrast, and color, as well as random cropping.
 * 
 * https://mxnet.apache.org/versions/1.5.0/tutorials/python/types_of_data_augmentation.html
 * 

 */
public final class ImageUtils {

    private ImageUtils() {
        // no instances of this class
    }    

    /**
     * Generates image with randomly filled rectangle, centered on the given dimensions. 
     * 
     * @param width
     * @param height
     * @param numberOfImages
     * @param destPath
     * @throws IOException 
     */
    static void generateRandomFilledCenteredRect(int width, int height, int numberOfImages, String destPath) throws IOException {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        
        Graphics gr = image.getGraphics();
        float r, g, b;
        
        Random rand = new Random();   
        for (int i = 0; i < numberOfImages; i++) {
            gr.setColor(Color.white);
            gr.fillRect(0, 0, width, height);
        
            r = (float) Math.random();
            g = (float) Math.random();
            b = (float) Math.random();

            int x = (int)( (width * 0.2) + rand.nextInt((int)(width * 0.2)));
            int rectWidth = (int)(2 * (0.5*width-x)); // rand.nextInt(width - x);
            gr.setColor(new Color(r, g, b));
            gr.fillRect(x, 0, rectWidth, height);
            
            String fileName = "negative_"+i+".jpg";

            ImageIO.write(image, "jpg", new File(destPath + "/" + fileName));
        }   
    }
  
    
    /**
     * Generates 4 translated variations of the given image: translates it for one third of an image width and height in each direction.
     * Uses black color to fill background extra space.
     * 
     * TODO: add param for background color or pattern to use to fill the empty space.
     * 
     * @param img image to translate
     * @return 
     */
    public static List<BufferedImage> translateImage(BufferedImage img) {
        List<BufferedImage> augImages = new ArrayList<>(); 
        AffineTransform[] trans = new AffineTransform[4];
        final int width = img.getWidth();
        final int height = img.getHeight();
        
            trans[0] = AffineTransform.getTranslateInstance(width/3, 0);
            trans[1] = AffineTransform.getTranslateInstance(-width/3, 0);
            trans[2] = AffineTransform.getTranslateInstance(0, height/3);
            trans[3] = AffineTransform.getTranslateInstance(0, -height/3);            
            
            for (int t = 0; t < 4; t++) {
                BufferedImage newImage = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
                Graphics2D dispGc = newImage.createGraphics();
                dispGc.setBackground(Color.BLACK); // enable to specify bg color, as a param
                dispGc.clearRect(0, 0, newImage.getWidth(), newImage.getHeight());
                dispGc.drawImage(img, trans[t], null);
                augImages.add(newImage);
            }
        
        return augImages;        
    }    
            
    /**
     * Scales specified image to given size and returns new image with specified width and height.
     * Original image is not changed..
     * 
     * @param img image to sclae
     * @param newWidth width of scaled image
     * @param newHeight height of scaled image
     * @return 
     */
    public static BufferedImage scaleImage(BufferedImage img, int newWidth, int newHeight) {    
        Image scaledImg = img.getScaledInstance(newWidth, newHeight, Image.SCALE_FAST);        
        BufferedImage resultImg = new BufferedImage(newWidth, newHeight, img.getType()); // BufferedImage.TYPE_INT_ARGB  img.getType()
        resultImg.getGraphics().drawImage(scaledImg, 0, 0, null);
                    
        return resultImg;        
    }
    
    /**
     * Scales input image to specified target width or height, centers and returns resulting image.
     * Scaling factor is calculated using larger dimension (width or height).
     * Keeps aspect ratio and image type, and bgColor parameter to fill background. 
     * 
     * @param img image to scale
     * @param targetWidth witdth to scale to
     * @param targetHeight height to scale to
     * @param padding minimalno rastojanje od ivica slike koja se smanjuje i centrira
     * @param bgColor, umesto bgColor bolje da bude neki pattern koji popinjava tipa random pixels
     * @return scaled and centered image
     */
    public static BufferedImage scaleAndCenter(BufferedImage img, int targetWidth, int targetHeight, int padding, Color bgColor) {
        final int imgWidth = img.getWidth();
        final int imgHeight = img.getHeight();
                
        float scaleFactor = 0;  // koliko treba smanjiti/skalirati sliku, faktor skaliranja
        int xPos, yPos;
        
        // scale by larger dimension
        if (imgWidth > imgHeight) {
            scaleFactor = imgWidth / (float)(targetWidth-2*padding);
        } else { // imgHeight < imgWidth
            scaleFactor = imgHeight / (float)(targetHeight-2*padding);
        }

        int newWidth = (int) (imgWidth / scaleFactor);
        int newHeight = (int)(imgHeight / scaleFactor);
        
        Image scaledImg = img.getScaledInstance(newWidth, newHeight, imgWidth);
        
        BufferedImage resultImg = new BufferedImage(targetWidth, targetHeight, img.getType());
        resultImg.getGraphics().setColor(bgColor);
        resultImg.getGraphics().fillRect(0, 0, targetWidth, targetHeight);
                
        if (imgWidth > imgHeight) {
            xPos = padding;
            yPos = padding + (targetHeight-2*padding - newHeight) / 2;            
        } else {
            xPos = padding + (targetWidth -2*padding - newWidth) / 2;
            yPos = padding;                                    
        }
        
        resultImg.getGraphics().drawImage(scaledImg, xPos, yPos, null);
                    
        return resultImg;
    }

    /**
     * Scales input image to specified target width or height, crops and returns resulting image.
     * Scaling factor is calculated using smaller dimension (width or height).
     * Keeps aspect ratio and image type, and bgColor parameter to fill background. 
     * 
     * @param img image to scale
     * @param targetWidth target image width
     * @param targetHeight target image height
     * @return scaled and cropped image
     */
    public static BufferedImage scaleBySmallerAndCrop(BufferedImage img, int targetWidth, int targetHeight) {

        int imgWidth = img.getWidth();
        int imgHeight = img.getHeight();
                
        float scale = 0;
        
        if (imgWidth < imgHeight) { // which one is smaller width or height?
            scale = imgWidth / (float)targetWidth; // scale by width
        } else { // imgHeight < imgWidth // scale by height
            scale = imgHeight / (float)targetHeight;
        }

        int newWidth = (int) (imgWidth / scale);
        int newHeight = (int)(imgHeight / scale);
        
        Image scaledImg = img.getScaledInstance(newWidth, newHeight, imgWidth);
               
        BufferedImage scaledBuffImg = new BufferedImage(newWidth, newHeight, img.getType());
        scaledBuffImg.getGraphics().drawImage(scaledImg, 0, 0, null);
        
        BufferedImage resultImage = null;
        
        if (imgWidth < imgHeight) { // crop by centering on  height
            final int xPos = 0;
            final int yPos = (newHeight - targetHeight) / 2;            
            resultImage = scaledBuffImg.getSubimage(xPos, yPos, targetWidth, targetHeight);            
        } else { // crop by centering on height
            final int xPos = (newWidth-targetWidth) / 2;
            final int yPos = 0;                                    
            resultImage = scaledBuffImg.getSubimage(xPos, yPos, targetWidth, targetHeight);            
        }
        
        return resultImage;
    }
    
    public static BufferedImage scaleBySmallerTarget(BufferedImage img, int targetWidth, int targetHeight) {

        int imgWidth = img.getWidth();
        int imgHeight = img.getHeight();
                
        float scale = 0;
        
        if (targetWidth < targetHeight) { // which one is smaller width or height?
            scale = imgWidth / (float)targetWidth; // scale by width
        } else { // imgHeight < imgWidth // scale by height
            scale = imgHeight / (float)targetHeight;
        }

        int newWidth = (int) (imgWidth / scale);
        int newHeight = (int)(imgHeight / scale);
        
        Image scaledImg = img.getScaledInstance(newWidth, newHeight, imgWidth);
               
        BufferedImage scaledBuffImg = new BufferedImage(newWidth, newHeight, img.getType());
        scaledBuffImg.getGraphics().drawImage(scaledImg, 0, 0, null);
        
        return scaledBuffImg;
    }    
    
    public static BufferedImage flipHorizontal(BufferedImage img) {
        BufferedImage resultImg = new BufferedImage(img.getWidth(), img.getWidth(), img.getType());
        final int width = img.getWidth(), height = img.getHeight();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                final int rgb = img.getRGB(x, y);
                resultImg.setRGB(width-1-x, y, rgb);
            }
        }
        return resultImg;
    }

    public static BufferedImage flipVertical(BufferedImage img) {
        BufferedImage flippedImage = new BufferedImage(img.getWidth(), img.getWidth(), img.getType());
        final int width = img.getWidth(), height = img.getHeight();
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {            
                final int rgb = img.getRGB(x, y);
                flippedImage.setRGB(x, height-1-y, rgb);
            }
        }
        return flippedImage;
    }    
    
    /**
     * Loads all images from the specified directory, and returns them as a list.
     * 
     * @param dir
     * @return list of images as BufferedImage instances
     * @throws IOException 
     */
    public static List<BufferedImage> loadImagesFromDirectory(File dir) throws IOException {
        List<BufferedImage> imageList = new ArrayList<>();
        for (final File file : dir.listFiles()) {
            if (!file.isDirectory()) {
                BufferedImage img = ImageIO.read(file);
                imageList.add(img);
            } 
        }        
        return imageList;
    }
    
    
    /**
     * Loads JPG, JPEG or PNG images from specified directory and returns them as a map 
     * with File object as a key and BufferedImage object as a value.
     * Skips all subdirectories.
     *  
     * @param dir directory to load
     * @return map of files and images.
     * @throws IOException 
     */
    public static HashMap<File, BufferedImage> loadFileImageMapFromDirectory(File dir) throws IOException {
        if (!dir.isDirectory()) throw new IllegalArgumentException("Parameter dir must be a directory: "+dir.toString());
        
        HashMap<File, BufferedImage> imageMap = new HashMap<>();
        for (final File file : dir.listFiles()) {
            if (file.isDirectory()) continue;// skip subdirectories
            
            final String imgType = getImageType(file);                       
            if (!imgType.equalsIgnoreCase("jpg") && !imgType.equalsIgnoreCase("jpeg") && !imgType.equalsIgnoreCase("png")) continue;
                                        
            BufferedImage img = ImageIO.read(file);
            imageMap.put(file, img);            
        }        
        return imageMap;
    }    
    
    /**
     * Returns the extension of the given image file.
     * 
     * @param file
     * @return 
     */
    public static String getImageType(final File file) {
            final String fileName = file.getName();
            return fileName.substring(fileName.lastIndexOf(".")+1); // get file ext/img type                       
    }
    
    /**
     * Returns grayscale version of the given image.
     * Original image is not changed.
     * 
     * @param img
     * @return 
     */
    public static BufferedImage grayscale(BufferedImage img) {
        BufferedImage resultImage = new BufferedImage(img.getWidth(), img.getWidth(), img.getType());
        final int width = img.getWidth(), height = img.getHeight();
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {            
                final int rgb = img.getRGB(x, y);
                final int red = ColorUtils.getRed(rgb);
                final int blue = ColorUtils.getBlue(rgb);
                final int green = ColorUtils.getGreen(rgb);
                final int alpha = 255;
                int gray = (red+green+blue)/3;                
                resultImage.setRGB(x, y, (new Color(gray, gray, gray, alpha)).getRGB());
            }
        }
        return resultImage;        
    }
    
    
    // hardcoded for lego replace with DataSetUtils
    /**
     * 
     * @param imageMap
     * @param imageFile
     * @param useAbsPath
     * @throws IOException 
     */
    public static void createIndexFile(HashMap<File, BufferedImage> imageMap, String imageFile, boolean useAbsPath) throws IOException {
        // dodaj regex pomocu koga ce da ih ubacuje uklas?
        try (BufferedWriter out = new BufferedWriter(new FileWriter(imageFile))) {
            int fileCount = imageMap.size();
            int i = 0;

            for (File file : imageMap.keySet()) {
                if (!useAbsPath) 
                    out.write(file.getName() + " legoman"); // TODO: how to get image label? - use parent folder name
                else
                    out.write(file.getPath() + " legoman");
                
                if (i < fileCount - 1) {
                    out.write(System.lineSeparator());
                }
                i++;
            }
        }
    }
    
    
    
    
    /**
     *  Returns an array of images created by translating specified input image 
     *  by specified number of count with specified step size.
     * 
     * @param img image to translate
     * @param stepCount number of 
     * @param stepSize 
     * @return 
     */
    public static BufferedImage[] randomTranslateImage(BufferedImage img, int stepCount, int stepSize) {
        // u krug za dati radijus
        // random distance
        BufferedImage[] images = new BufferedImage[stepCount*4];  
        AffineTransform[] trans = new AffineTransform[4];
        
        for(int i = 0; i < stepCount; i++) {
            // pomeri na sve cetiri strane po count * step pixela
            trans[0] = AffineTransform.getTranslateInstance(i*stepSize, 0);
            trans[1] = AffineTransform.getTranslateInstance(-i*stepSize, 0);
            trans[2] = AffineTransform.getTranslateInstance(0, i*stepSize);
            trans[3] = AffineTransform.getTranslateInstance(0, -i*stepSize);            
            
            for (int t = 0; t < 4; t++) {
                BufferedImage newImage = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
                Graphics2D dispGc = newImage.createGraphics();
                dispGc.setBackground(Color.WHITE);
                dispGc.clearRect(0, 0, newImage.getWidth(), newImage.getHeight());
                dispGc.drawImage(img, trans[t], null);
                images[i*4+t] = newImage;
            }
        }
        
        return images;        
    }
    

    /**
     * Crops specified number of random sub-images with specified dimensions.
     * 
     * @param img image to crop
     * @param cropWidth width of the cropped image
     * @param cropHeight height of the cropped image
     * @param cropNum number of images to crop
     * @param random random number generator used to generate random positions
     * @return list of randomly cropped images
     */
    public static List<BufferedImage> randomCrop(BufferedImage img, int cropWidth, int cropHeight, int cropNum, Random random) {
        List<BufferedImage> croppedImages = new ArrayList<>(cropNum);
        
        if (cropWidth > img.getWidth()) throw new RuntimeException("Crop width is larger then image width!");
        if (cropHeight > img.getHeight()) throw new RuntimeException("Crop height is larger then image height!");
        
        final int maxX = img.getWidth() - cropWidth;
        final int maxY = img.getHeight() - cropHeight;
        
        int x, y;
        
        for(int i=0; i< cropNum; i++) {
            if (maxX > 5) x = random.nextInt(maxX);
                else x = 0;
            
            if (maxY > 5) y = random.nextInt(maxY);
                else y = 0;
            
            BufferedImage cropped = img.getSubimage(x, y, cropWidth, cropHeight);
            croppedImages.add(cropped);
        }
        
        return croppedImages;
    }
    
    
    /**
     * Returns five crops of a given image: four from each corner and one at center
     * 
     * @param img image to crop
     * @param cropWidth width of the cropped area
     * @param cropHeight height of the cropped area
     * @return five crops from the corners and center of the image
     */
    public static List<BufferedImage> cropAtCornersAndCenter(BufferedImage img, int cropWidth, int cropHeight) {
        List<BufferedImage> croppedImages = new ArrayList<>(5);
        
        int imgWidth = img.getWidth();
        int imgHeight = img.getHeight();
        
        // check if specified crop dimensions are larger then the image
        if (cropWidth > imgWidth) throw new RuntimeException("Crop width is larger then image width!");
        if (cropHeight > imgHeight) throw new RuntimeException("Crop height is larger then image height!");
        
        // top left crop
        BufferedImage crop = img.getSubimage(0, 0, cropWidth, cropHeight);
        croppedImages.add(crop);
        
        // top right crop
        crop = img.getSubimage(imgWidth - cropWidth-1, 0 , cropWidth, cropHeight);
        croppedImages.add(crop);        

        // bottom left crop
        crop = img.getSubimage(0, imgHeight - cropHeight - 1 , cropWidth, cropHeight);
        croppedImages.add(crop);        
        
        // bottom right crop
        crop = img.getSubimage(imgWidth - cropWidth-1, imgHeight - cropHeight - 1 , cropWidth, cropHeight);
        croppedImages.add(crop);                
        
        // center crop - fin middle width/2 - crop/2                
        int x, y;        
        
        x = imgWidth / 2 - cropWidth / 2;
        y = imgHeight / 2 - cropHeight / 2;
                  
        crop = img.getSubimage(x, y, cropWidth, cropHeight);
        croppedImages.add(crop);
        
        
        return croppedImages;
    }
    
    
    /**
     * Randomly crop image with specified width and height.
     * Returns cropped image, while original image is unchanged.
     * 
     * @param img
     * @param cropWidth
     * @param cropHeight
     * @return cropped image
     */
    public static BufferedImage randomCrop(BufferedImage img, int cropWidth, int cropHeight) {      
        // ovo treba  samo ako je slika veca od zadatih crop dimenzija ako nije onda je uzmi celu
        // mozda dodati i random crop koji cropuje tacno 5 slika, coskove i random centar
        // proveriti dimenzije slike
        final int maxX = img.getWidth() - cropWidth;
        final int maxY = img.getHeight() - cropHeight;
        Random random = RandomGenerator.getDefault().getRandom();
        final int x = random.nextInt(maxX);
        final int y = random.nextInt(maxY);
        BufferedImage cropped = img.getSubimage(x, y, cropWidth, cropHeight);

        return cropped;
    }    
    
    
    //
    /**
     * Still not working as it should
     * @param img input image
     * @param maxTint maximum tint value
     * @param maxBrightness maximum brightness value
     * @param num number of variations
     * @param random random generator
     * @return list of augmented images
     * @see <a href="https://en.wikipedia.org/wiki/Normalization_(image_processing)">Image Normalization</a>
     */
    public static List<BufferedImage> randomTintAndBrightness(BufferedImage img, float maxTint, int maxBrightness, int num, Random random) {
        // hinton mnozi sopstene vektore i dodaje ih na asliku ima objasnjeno i umagenet radu
        List<BufferedImage> augmentedImages = new ArrayList<>();
        // nekako klimavo ali nesto radi samo sa dodavanjem 
        for(int i=0; i< num; i++)  { // for all images, probably should bedone i parallel
            final float k = random.nextFloat()*maxTint;
            final int n = random.nextInt(maxBrightness);
         
            for (int y = 0; y < img.getHeight(); y++) { // process all pixels
                for (int x = 0; x < img.getWidth(); x++) {
                    int rgb = img.getRGB(x, y);
                    //rgb = (int)(  k*rgb);
                    rgb = rgb+n;
                    if (rgb>255) rgb=255;
                    img.setRGB(x, y, rgb);
                }
            }       
            BufferedImage newImg= deepCopy(img);
            augmentedImages.add(newImg);            
        }
        
        return augmentedImages;
    }
    
    //TODO: razvlacenje kontrasta, thresholding, edge detection filters
    
    public static BufferedImage randomTintAndBrightness(BufferedImage img) {
        Random random = new Random();
        BufferedImage resultImage = new BufferedImage(img.getWidth(), img.getWidth(), img.getType());
        //         dstElement = (srcElement*scaleFactor) + offset
            final float k = 0.5f + random.nextFloat()*0.75f;
            final int n = random.nextInt(255); // 0.75 - 1.25

            for (int y = 0; y < img.getHeight(); y++) { // process all pixels
                for (int x = 0; x < img.getWidth(); x++) {
                    int rgb = img.getRGB(x, y);
                    Color c = new Color(rgb);
                    int red =(int) (ColorUtils.getRed(rgb)*k);
                    if (red>255) red = 255;
                    int blue = (int)(ColorUtils.getBlue(rgb)*k);
                    if (blue>255) blue = 255;
                    int green = (int)(ColorUtils.getGreen(rgb)*k);                    
                    if (green>255) green = 255;
                    int alpha =(int)(c.getAlpha() * 0.8);
                    if (alpha>255) alpha = 255;
                    Color newColor = new Color(red, green, blue, alpha);
                    resultImage.setRGB(x, y, newColor.getRGB());
                }
            }               
                
        return resultImage;
    }    
    
    
    static BufferedImage deepCopy(BufferedImage bi) {
        ColorModel cm = bi.getColorModel();
        boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
        WritableRaster raster = bi.copyData(null);
        
        // https://stackoverflow.com/questions/3514158/how-do-you-clone-a-bufferedimage
      //  return new BufferedImage(cm, raster, isAlphaPremultiplied, null).getSubimage(0, 0, bi.getWidth(), bi.getHeight());
         return new BufferedImage(cm, raster, isAlphaPremultiplied, null); 
    }    
    
    /**
     * Writes list of images to specified file path.
     * 
     * @param images images to write
     * @param targetPath path to write to
     * @param fileNamePrefix prefix of image file name to use (eg. for prefix pattern "someImage" it will create file someImage_1.jpg )
     * @param fileType type of file/images to write
     * @throws IOException 
     */
    public static void writeImages(List<BufferedImage> images, String targetPath, String fileNamePrefix, String fileType) throws IOException {
        int i=0;
        for(BufferedImage img : images) {
            i++;
            ImageIO.write(img, fileType, new File(targetPath + File.separator  +fileNamePrefix+"_"+i+"."+fileType));
        }
    }
    
    
    /**
     * Generates a specified number of randomly full colored images of a specified size.
     * Images are saved at specified destination path named using negative_x.jpg name pattern
     * 
     * @param width image width
     * @param height image height
     * @param numberOfImages number of images to generate
     * @param destPath destination path to save images
     * 
     * @throws IOException 
     */
    public static void generateRandomColoredImages(int width, int height, int numberOfImages, String destPath) throws IOException{
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        
        Graphics gr = image.getGraphics();
        float r, g, b;
        
        for (int i = 0; i < numberOfImages; i++) {
            r = (float) Math.random();
            g = (float) Math.random();
            b = (float) Math.random();

            gr.setColor(new Color(r, g, b));
            gr.fillRect(0, 0, width, height);
            String fileName = "negative_"+i+".jpg";

            ImageIO.write(image, "jpg", new File(destPath + "/" + fileName));
        }        
    }
    
    public static void generateRandomFilledRectImages(int width, int height, int numberOfImages, String destPath) throws IOException{
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        
        Graphics gr = image.getGraphics();
        float r, g, b;
        
        Random rand = new Random();
        
        for (int i = 0; i < numberOfImages; i++) {
            r = (float) Math.random();
            g = (float) Math.random();
            b = (float) Math.random();

            gr.setColor(new Color(r, g, b));
            gr.fillRect(0, 0, width, height);
            
            for(int k=0; k<10; k++) {
                r = (float) Math.random();
                g = (float) Math.random();
                b = (float) Math.random();
                
                int x = rand.nextInt((int)(width*0.75));
                int y = rand.nextInt((int)(height*0.75));
                int w =  rand.nextInt(width-x);
                int h =  rand.nextInt(height-y);
                gr.setColor(new Color(r, g, b));
                gr.fillRect(x, y, w, h);
            }
            
            
            String fileName = "negative_"+i+".jpg";

            ImageIO.write(image, "jpg", new File(destPath + "/" + fileName));
        }        
    }    
    
    public static void generateNoisyImage(int width, int height, int numberOfImages, String destPath) throws IOException {
        //create buffered image object img
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        float r, g, b;
        //file object
        for (int i = 0; i < numberOfImages; i++) {
            //create random image pixel by pixel
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    r = (float) Math.random(); //red
                    g = (float) Math.random();; //green
                    b = (float) Math.random(); //blue

                    Color col = new Color(r, g, b);
                    image.setRGB(x, y, col.getRGB());
                }
            }
            String fileName = "noisy_" + i + ".jpg"; // 

            ImageIO.write(image, "jpg", new File(destPath + "/" + fileName));
        }

    }    
    
   
    
    
}
