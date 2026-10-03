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


/**
 * Koristi se verovatno u labeleru.
 * 
 */
public final class BoundingBox {
    private int id; // bb id
    private int xCenter, yCenter, width, height;
    private int xMin, yMin, xMax, yMax;//  da bude cao mscocoi pascal voc
    private int classId;
    private String label; // class label
    //private List<String> classes; // odavde moze da izme label za class id
    // dodaj classId i mozda listu svih klasa da ucitava direktno iz coco i voc formata
    // kako ga uvezati sa ExampliImage da sadrzi listu BoundingBoxes kad ucitava

    private float score;    // ovo kad prepoznaje kod ucitavanja je nula
    

    public BoundingBox(int id, int xCenter, int yCenter, int width, int height, float score) {
        this.id = id;
        this.xCenter = xCenter;
        this.yCenter = yCenter;
        this.width = width;
        this.height = height;
        this.score = score;
    }
    
    public BoundingBox(int xCenter, int yCenter, int width, int height) {
        this.xCenter = xCenter;
        this.yCenter = yCenter;
        this.width = width;
        this.height = height;
    }
    
    // xMin, xMax, yMin, yMax
    
    public BoundingBox(int id, int xCenter, int yCenter, int width, int height) {
        this.id =id;
        this.xCenter = xCenter;
        this.yCenter = yCenter;
        this.width = width;
        this.height = height;
    }    
    
    public BoundingBox(int id, int xCenter, int yCenter, int width, int height, String label, float score) {
        this.id = id;
        this.xCenter = xCenter;
        this.yCenter = yCenter;
        this.width = width;
        this.height = height;
        this.label = label;
        this.score = score;
    }    

    public int getYCenter() {
        return yCenter;
    }

    public void setYCenter(int yCenter) {
        this.yCenter = yCenter;
    }

    public int getXMin() {
        return xMin;
    }

    public void setXMin(int xMin) {
        this.xMin = xMin;
    }

    public int getYMin() {
        return yMin;
    }

    public void setYMin(int yMin) {
        this.yMin = yMin;
    }

    public int getYMax() {
        return xMax;
    }

    public void setXMax(int xMax) {
        this.xMax = xMax;
    }

     public void setYMax(int yMax) {
        this.yMax = yMax;
    }
    
    
    

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public float getScore() {
        return score;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getLabel() {
        return label;
    }
    
    

    public void setScore(float score) {
        this.score = score;
    }

    public void setId(int id) {
       this.id = id;
    }    
    
    public int getId() {
        return id;
    }

    @Override
    public String toString() {
        return "BoundingBox{" + "id=" + id + ", xCenter=" + xCenter + ", y=" + yCenter + ", width=" + width + ", height=" + height + ", label=" + label + ", score=" + score + '}';
    }


    
    
    
    


}
