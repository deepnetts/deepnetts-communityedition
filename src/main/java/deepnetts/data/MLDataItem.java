package deepnetts.data;

import deepnetts.tensor.TensorBase;

/**
 * Single data item that will be used to train machine learning model. 
 * Implementing classes should provides methods for input and target output for a machine learning model.
 * To create a data set of custom objects implement this interface for corresponding class.
 * 
 * @see DataSet
 */
public interface MLDataItem { // <I, O>

    /**
     * Returns an input for machine learning model of this item.
     * @return an example input for a machine learning model training.
     */
    public TensorBase getInput();
    
    /**
     * Returns target output for machine learning model of this item.
     * @return target output 
     */
    public TensorBase getTargetOutput();
    
    /**
     * Returns a model output error for this data item. 
     * @return 
     */
    public float getError(); //  ja bih zapravo trebalo da uzmem vrednost koja je dodata na ukupni loss za ovaj patern to je jedna vrednost a ane ceo niz
    public void setError(float error);
}
