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

package deepnetts.data;

import deepnetts.util.ImageUtils;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.TensorBase;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * Example image to train a deep learning model.
 * 
 * It holds image pixels and corresponding label.
 */
public class ExampleImage implements MLDataItem {

    /**
     * Image dimensions - width and height
     */
    private final int width, height;   // dont need this here , maybe only in dataset

    /**
     * Image label, a concept to map to this image
     */
    private final String label;

    /**
     * Desired network output - maybe its better to use  int - output index with 1 ? lesss memory for huge data sets - TODO: use int here
     */
    private TensorBase targetOutput; // output vector depends on number of classes- this could be int in order to save memory

    private float error;
    
    /**
     * Transformed RGB values of Image pixels
     * used as an input for neural net
     */
    private float[] rgbVector;

    protected Tensor3D rgbTensor;

 //   private BufferedImage image;
    private File file;

    public ExampleImage(String imgFile, String label) throws IOException {
        this(new File(imgFile), label);
    }    
    
    /**
     * Creates an instance of new example image with specified image and label
     * Loads image from specified file and creates matrix structures with color information
     *
     * @param imgFile image file
     * @param label image label
     * @throws IOException if file is not found or reading file fails from some reason.
     */
    public ExampleImage(File imgFile, String label) throws IOException {
        this.label = label;
        this.file = imgFile;
        BufferedImage image = ImageIO.read(imgFile);
        width = image.getWidth();
        height = image.getHeight();

        createInputFromPixels(image, 3);
    }

    public ExampleImage(BufferedImage image, String label) {
       // this.image = image;
        this.label = label;
        width = image.getWidth();
        height = image.getHeight();

        createInputFromPixels(image, 3);
    }
    
    public ExampleImage(BufferedImage image, String label, int channels) {
       // this.image = image;
        this.label = label;
        width = image.getWidth();
        height = image.getHeight();

        createInputFromPixels(image, channels);
    }    
    
    public ExampleImage(File originalImgFile, BufferedImage image, String label) {
        this(image, label);
        this.file = originalImgFile;
    }
    
    public ExampleImage(BufferedImage image) {
        this(image, null);
    }    
    
    public ExampleImage(BufferedImage image, String label, int targetWidth, int targetHeight) throws IOException {
      //  this.image = image;
        this.label = label;
        width = targetWidth;
        height = targetHeight;  
        
        // if specified image does not fit given dimsnsions scale image
        if (image.getWidth() != targetWidth || image.getHeight() != targetHeight) {
            image = ImageUtils.scaleImage(image, targetWidth, targetHeight);
        }
      
        createInputFromPixels(image, 3);
    }    
    
    protected void createInputFromPixels(BufferedImage image, int channels) {
        rgbVector = new float[width * height * channels];

        // ako image nije sRGB
        if (image.getType() != BufferedImage.TYPE_INT_ARGB) {
            BufferedImage imageCopy = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_ARGB);
            imageCopy.getGraphics().drawImage(image, 0, 0, null);
            image = imageCopy;
        }
        Raster raster = image.getRaster();
        float[] pixel = null;

        // fill tensor from pixels (this should be in NCHW or NHWC format?) @fixlayout
        // pa ovaj je vec u chw? rzmisli u kom je ovaj formatu? N trenutno nema
        // ovako je bilo
        // nchw row major
//        for (int y = 0; y < height; y++) { // samo zameni redosled ova dva i dole index za rgb
//            for (int x = 0; x < width; x++) {
//                pixel = raster.getPixel(x, y, pixel); // get as butes
//
//                rgbVector[y * width + x] = pixel[0] / 255.0f; // ovo je row major 
//                if (channels>1) {
//                    rgbVector[width * height + y * width + x] = pixel[1] / 255.0f;
//                    rgbVector[2 * width * height + y * width + x] = pixel[2] / 255.0f;
//                }
//            }
//        }
        
        // ovo je column major NCHW
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) { // samo zameni redosled ova dva i dole index za rgb
                pixel = raster.getPixel(x, y, pixel); // get as butes

                rgbVector[x * height + y] = pixel[0] / 255.0f; // ovo je column major  HW
                if (channels>1) {
                    rgbVector[width * height + x * height + y] = pixel[1] / 255.0f; // ovo je chw layout
                    rgbVector[2 * width * height + x * height + y] = pixel[2] / 255.0f;
                }
            }
        }

        rgbTensor = new Tensor3D(channels, height, width, rgbVector); 
    }
    
    public void invert() {
        for (int i = 0; i < rgbVector.length; i++) {
            rgbVector[i] = 1 - rgbVector[i];
        }
    }

    @Override
    public TensorBase getTargetOutput() {
        return targetOutput;
    }

    public float[] getRgbVector() {
        return rgbVector;
    }

    // set this internally using utility method
    public final void setTargetOutput(TensorBase targetOutput) {
        this.targetOutput = targetOutput;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public String getLabel() {
        return label;
    }

    @Override
    public Tensor3D getInput() {
        return rgbTensor;
    }

    public File getFile() {
        return file;
    }
    
    public BufferedImage getOriginalImageFromFile() throws IOException {
        if (file == null) {
            throw new IllegalStateException("Image file has never been set in previous usage!");
        }
        
        return ImageIO.read(file);
    }

//    public BufferedImage getImage() {
//        return image;
//    }

    @Override
    public float getError() {
        return error;
    }

    @Override
    public void setError(float error) {
        this.error = error;
    } 

   

}
