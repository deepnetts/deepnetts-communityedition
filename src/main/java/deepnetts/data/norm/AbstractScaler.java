package deepnetts.data.norm;

import deepnetts.tensor.TensorBase;
import java.io.Serializable;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;
import javax.visrec.ml.data.preprocessing.Scaler;

/**
 * Base class to simplify implementation of custom normalization procedure.
 * Extend this class to cretae new normalizaton procedures.
 * 
 */
public abstract class AbstractScaler implements Scaler<DataSet<MLDataItem>>, Serializable {
    
    /**
     * Normalize input of  deployed model
     * 
     * @param input 
     */
    public abstract void scaleInput(TensorBase input);       
    
}

