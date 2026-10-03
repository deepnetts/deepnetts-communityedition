package deepnetts.util;

import deepnetts.tensor.TensorBase;
import deepnetts.data.Preprocessing;
import deepnetts.tensor.Tensor;
import java.io.Serializable;


/**
 * Image pre-processing to be used at inference time, after feeding input to the network.
 */
public class ImagePreprocessing implements Preprocessing<Tensor>, Serializable {
    private boolean subMean=false;  
    private boolean invertPixels=false;
    private boolean scaleImage=false;
    private TensorBase mean;
    private boolean isEnabled=false;
    
    // mozda konvolucionoj da dodam setInput(BufferedImage), a sta  sa Bitmapom?
    // to treba da se desi u exImage-u mozda najbolje sve prebaciti tamo?
    @Override
    public void apply(Tensor input) {
        if (!isEnabled) return;
        
        // if invert: 1-pix
        if (invertPixels) {            
            float[] values = input.getValues();
            for(int i=0; i<values.length; i++) {
                values[i] = 1 - values[i];
            }
        }
        
        // how to scale image - must be done before this - probably in image Classifier        
        
        // sub mean 
        if (subMean) {
            ((TensorBase)input).sub(mean);
        }        
    }

    public boolean isSubMean() {
        return subMean;
    }

    public void setSubMean(boolean subMean) {
        this.subMean = subMean;
    }

    public boolean isInvertPixels() {
        return invertPixels;
    }

    public void setInvertPixels(boolean invertPixels) {
        this.invertPixels = invertPixels;
    }

    public TensorBase getMean() {
        return mean;
    }

    public void setMean(TensorBase mean) {
        this.mean = mean;
    }

    public boolean isEnabled() {
        return isEnabled;
    }

    public void setEnabled(boolean enabled) {
        this.isEnabled = enabled;
    }
             
}
