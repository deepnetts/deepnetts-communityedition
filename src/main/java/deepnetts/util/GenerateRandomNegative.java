package deepnetts.util;

import java.io.IOException;

public class GenerateRandomNegative {
    public static void main(String[] args) throws IOException { 
     // ImageUtils.generateRandomColoredImages(64,64, 66, "D:/datasets/DukeSet/negative");
      //ImageUtils.generateRandomFilledRectImages(64,64, 20, "C:/Users/Zoran/Documents/DukeSet/negative");
      ImageUtils.generateRandomFilledCenteredRect(64,64, 20, "C:/Users/Zoran/Documents/DukeSet/negative");
     //ImageUtils.generateNoisyImage(64, 64, 30, "C:/Users/Zoran/Documents/DukeSet/negative"); // 
      //  ImageSetUtils.createImageIndex("D:\\datasets\\LegoPeopleNoviJecaPreprocessed\\");
    }
}
