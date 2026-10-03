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

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import javax.imageio.ImageIO;

/**
 * Center images on backgounds and save at target path.
 *

 */
public class ObjectsOnBackgrounds {

    public static void main(String[] args) throws IOException {
        String objectSourcePath = "/home/zoran/Desktop/JavaOneSponsors/logos";
        String bgSourcePath = "/home/zoran/Desktop/JavaOneSponsors/bg";
        String targetPath = "/home/zoran/Desktop/JavaOneSponsors/centered";

        HashMap<File, BufferedImage> objects = ImageUtils.loadFileImageMapFromDirectory(new File(objectSourcePath));
        HashMap<File, BufferedImage> backgrounds = ImageUtils.loadFileImageMapFromDirectory(new File(bgSourcePath));

        Iterator<Entry<File, BufferedImage>> objIter = objects.entrySet().iterator();

        while (objIter.hasNext()) { // iterate all object images
            Entry<File, BufferedImage> objEntry = objIter.next();
            BufferedImage objImg = objEntry.getValue();
            String objName = objEntry.getKey().getName();

            String imgType = objName.substring(objName.indexOf(".")+1);
            
            Iterator<Entry<File, BufferedImage>> bgIter = backgrounds.entrySet().iterator();
            

            while (bgIter.hasNext()) { // iterate all backgrounds
                BufferedImage bgImg = bgIter.next().getValue(); //ovde je problem sto on odmah sve jedne preko drugih

                bgImg.getGraphics().drawImage(objImg, 0, 0, null);
                ImageIO.write(bgImg, imgType, new File(targetPath + "/" + objName));  //  TODO: set image type

            }
        }
    }

}
