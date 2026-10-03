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
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 * just move 2(x) pix to left right up down
 * 

 */
public class DeleteAugmentImages {
    
    String sourcePath = "D:\\datasets\\DukeSet\\duke\\";
    String targetPath = "D:\\datasets\\DukeSet\\augmented\\"; 
    
    String imageIndexFileFile ="";
    String labelsFile ="";
    
    public void run() {
        try {
            HashMap<File, BufferedImage> images = ImageUtils.loadFileImageMapFromDirectory(new File(sourcePath));
                                   
            int i = 0;
            for(File file : images.keySet()) {
              //  if (!isImageFile(file)) continue;
           
                String fileName = file.getName();
                if (fileName.contains("_aug_")) file.delete();
                i++;
            }

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
        DeleteAugmentImages demo = new DeleteAugmentImages();
        demo.run();                
    }
}
