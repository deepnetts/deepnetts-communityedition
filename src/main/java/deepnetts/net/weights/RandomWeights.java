package deepnetts.net.weights;

import deepnetts.util.RandomGenerator;

/**
 * Weights randomization utility methods.
 */
public class RandomWeights {
    
    private static RandomGenerator randomGen =  RandomGenerator.getDefault();
    
    /**
     * Initializes random number generator with specified seed.
     * 
     * @param seed init value for random generator
     */
    public static void initSeed(long seed) {
        RandomGenerator.getDefault().initSeed(seed);
    }
        
    /**
     * Initialize the elements of specified array with random numbers with uniform distribution in range [-0.5, 0.5].
     *
     * @param array array to initialize
     */
    public static void randomize(float[] array) {
        for (int i = 0; i < array.length; i++) {
           array[i] = randomGen.nextFloat() - 0.5f;
        }
    }
  
        
    public static void widrowHoff(float[] array, float input, float hidden) {         
        randomize(array);
        float beta = 0.7f * (float)Math.pow(hidden, 1/input);   //1/input* 
        
        float weightsNorm =0;
        for (int i = 0; i < array.length; i++) {
            weightsNorm += array[i]*array[i];
        }
        weightsNorm = (float)Math.sqrt(weightsNorm);
        
        for (int i = 0; i < array.length; i++) {
            array[i] = (beta*array[i]) / weightsNorm;
        }
    }    
    
    
    /**
     * Uniform U[-a,a] where a=1/sqrt(in). 
     *   
     * Understanding the difficulty of training deep feedforward neural networks
     * Xavier Glorot, Yoshua Bengio, 2010, Commonly used "heuristic" , Eq. 1 pg 251
     * http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
     * http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
     * 
     * This method is reffered to as commonly used heuristics in paper above.
     * 
     * @param weights an array of weights to randomize
     * @param numInputs a number of inputs from previous layer
     */
    public static void uniform(float[] weights, int numInputs) {        
        if (numInputs==0) throw new IllegalArgumentException("Number of inputs for uniform randomization cannot be zero!");
        
        float min = -1 / (float)Math.sqrt(numInputs);
        float max = 1 / (float)Math.sqrt(numInputs);
      
        for (int i = 0; i < weights.length; i++) {
           weights[i] =  min + (randomGen.nextFloat()* (max-min));
        }        
    }
    
    public static void uniform(float[] weights, float min, float max) {              
       for (int i = 0; i < weights.length; i++) {
           weights[i] =  min + (randomGen.nextFloat()* (max-min));
       }             
    }
    

    /**
     * He initialization, used for relu activations.zero-mean. 
     * Zero-mean Gaussian distribution whose standard deviation (std) is√2/nl. (eq 10 in paper below)
     * Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
     * https://arxiv.org/pdf/1502.01852.pdf
     * Koristi za Relu i leaky relu
     * 
     * @param weights weights to initialize
     * @param numInputs number of inputs
     */
    public static void he(float[] weights, int numInputs) {      
        gaussian(weights, 0, (float)Math.sqrt(2.0f/numInputs));
    }    
    
    // Box–Muller transform
    // https://www.javamex.com/tutorials/random_numbers/gaussian_distribution_2.shtml
    // http://mathworld.wolfram.com/Box-MullerTransformation.html
    // https://stackoverflow.com/questions/19944111/creating-a-gaussian-random-generator-with-a-mean-and-standard-deviation    
      // 70% of values will be between -1 and 1
    public static void gaussian(float[] weights, float mean, float std) {      
       for (int i = 0; i < weights.length; i++) {
           weights[i] =  randomGen.nextGaussian()*std + mean;       
       }
    }   
    
    // 70% of values will be between -1 and 1
    // zero mean, 1 std
    public static void normal(float[] weights) {      
       gaussian(weights, 0, 1); 
    }     
    
    /**
     * Normalized uniform initialization U[-a,a] with a = sqrt(6/(in + out)).
     * Properly scaled uniform distribution (normalized initialization).
     * 
     * Sta treba u brojiocu 6 ili 1 u=i da li u imeniocu treba zbir in + out ili samo in ili da aimam obe verzije
     * i pitanje je da li je ispod uniformna ili gausova distribucija
     * 
     * Xavier Glorot, Yoshua Bengio, 2010, 
     * Understanding the difficulty of training deep feedforward neural networks
     * http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf  ,  pg. 253, eq 16.
     * http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
     * 
     * Use for tanh 
     * 
     * @param weights
     * @param numIn  size of the previous layer (number of inputs)
     * @param numOut size of initialized layer (number of outputs)
     * 
     * https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
     */
     public static void xavier(float[] weights, int numIn, int numOut) {
        float min = (float)-Math.sqrt( 6 / (float)(numIn+numOut));
        float max = (float)Math.sqrt( 6 / (float)(numIn+numOut));
        // zasto je ovde 6 / ? da li je ta vrednost fiksna best practice za sta???
        
        // ovo ne bi trebalo za celu matricu vec za svaki row da ima normalnu distribuciju skaliranu na normal
        for (int i = 0; i < weights.length; i++) {
           weights[i] =  min + (randomGen.nextFloat() * (max-min));
        }                       
    }
}
