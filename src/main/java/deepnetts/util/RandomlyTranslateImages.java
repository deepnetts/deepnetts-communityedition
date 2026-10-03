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
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 * just move 2(x) pix to left right up down
 * 

 */
public class RandomlyTranslateImages {
    
    String sourcePath = "/home/zoran/Desktop/JavaOneSponsors/";
    String targetPath = "/home/zoran/Desktop/JavaOneSponsors/translated/"; 
    
    String imageIndexFileFile ="";
    String labelsFile ="";
    
    public void run() {
        try (BufferedWriter br = new BufferedWriter(new FileWriter(targetPath + "translatedImages.txt")) ){
            HashMap<File, BufferedImage> images = ImageUtils.loadFileImageMapFromDirectory(new File(sourcePath));
                       
            
            int i = 0;
            for(File file : images.keySet()) {
                if (!isImageFile(file)) continue;
                BufferedImage[] translatedImages = ImageUtils.randomTranslateImage(images.get(file), 2, 2);
                String fileName = file.getName();
                String name = fileName.substring(0, fileName.lastIndexOf("."));
                String imgType = fileName.substring(fileName.lastIndexOf(".")+1);
                                
                
                for(int j=0; j<translatedImages.length; j++) {
                    String newFileName = name +"_"+j+ "."+imgType;
                    ImageIO.write(translatedImages[j], imgType, new File(targetPath + newFileName));
                    br.write(targetPath + newFileName + " " + name + System.lineSeparator());
                }
                i++;
            }

            br.close();
        } catch (IOException ex) {
            Logger.getLogger(RunScaleImages.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public boolean isImageFile(File file) {
        String fileName = file.getName();
        
        if (fileName.equalsIgnoreCase("white.jpg")) return false;
        
        String type = fileName.substring(fileName.lastIndexOf("."));
        if (!type.equalsIgnoreCase(".jpg") && !type.equalsIgnoreCase(".png")) return false;
        return true;
    }
    
    
    
    public static void main(String[] args) {
        RandomlyTranslateImages demo = new RandomlyTranslateImages();
        demo.run();                
    }
}
