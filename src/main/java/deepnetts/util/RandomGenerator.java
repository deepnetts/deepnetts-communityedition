package deepnetts.util;

import java.util.Random;

/**
 * Random number generator singleton.
 * 
 */
public class RandomGenerator {
    // verovatno ne bi trebalo singleton ako ima vise mreza itd.
    private static RandomGenerator instance;
    
    private Random randomGen =  new Random(123);
        
    /**
     * Prevent instantiation of this class
     */
    private RandomGenerator() { }
    
    /**
     * Returns the default global instance of random generator
     * @return 
     */
    public final static RandomGenerator getDefault() {
        if (instance == null) {
            instance = new RandomGenerator();
        }
        
        return instance;
    }
    
    /**
     * Returns underlying Java random generator
     * @return 
     */
    public Random getRandom() {
        return  randomGen;
    }    
    
    /**
     * Initialize the underlying Java random number generator with specified seed.
     * Can be also used to reinitialize the random number generator.
     * @param seed 
     */
    public void initSeed(long seed)  {
        randomGen = new Random(seed);
    }
    
    public float nextFloat() {
        return randomGen.nextFloat();
    }
    
    public float nextGaussian() {
        return (float)randomGen.nextGaussian();
    }    
    
    public int nextInt() {
        return randomGen.nextInt();
    }
    

    

}
