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

/**
 *

 */
public final class ColorUtils {

    private ColorUtils() { }

    public final static int getRed(final int color) {
        return (color >> 16) & 0xFF;
    }

    public final static int getGreen(final int color) {
        return (color >> 8) & 0xFF;
    }

    public final static int getBlue(final int color) {
        return color & 0xFF;
    }
    
    public static Color getColorFor(float min, float max, float val) { // value [0..1]
        // max alpha FF
//        def rgb(minimum, maximum, value):
//    minimum, maximum = float(minimum), float(maximum)
//    ratio = 2 * (value-minimum) / (maximum - minimum)
//    b = int(max(0, 255*(1 - ratio)))
//    r = int(max(0, 255*(ratio - 1)))
//    g = 255 - b - r
//    return r, g, b
        int r, g, b;        
        float ratio = 2 * (val-min) / ((float)(max-min));
        
        b = (int)Math.max(0, 255*(1-ratio));
        r = (int)Math.max(0, 255*(ratio-1));
        g = 255 - b - r;
        System.out.println(val);
        Color color = new Color(r, g, b, 255);
        return color;
    }
    
    
//    public int getIntFromColor(float Red, float Green, float Blue) {
//        int R = Math.round(255 * Red);
//        int G = Math.round(255 * Green);
//        int B = Math.round(255 * Blue);
//
//        R = (R << 16) & 0x00FF0000;
//        G = (G << 8) & 0x0000FF00;
//        B = B & 0x000000FF;
//
//        return 0xFF000000 | R | G | B;
//    }

}
