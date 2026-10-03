package deepnetts.util;

import java.util.Properties;

/**
 * Provides methods for getting typed properties for the given key.
 */
public class TypedProperties extends Properties {
        
   public TypedProperties(Properties prop) {
       
       for (Object key : prop.keySet()) {
            this.put(key, prop.get(key));
       }
   } 
    
    /**
     * Get specified property as int.
     * @param key
     * @return 
     */
    public int getInt(String key) {
        return Integer.parseInt(getProperty(key));
    }

    public int[] getIntArray(String key) {
        return (int[])get(key);
    }
    
    /**
     * Get specified property as float.
     * @param key name of the property
     * @return 
     */    
    public float getFloat(String key) {
        return Float.parseFloat(getProperty(key));
    }
    
    public Float[] getFloatArray(String key) {
        return (Float[])get(key);
    }    
    
    /**
     * Get specified property as double .
     * @param key name of the property
     * @return 
     */    
    public double getDouble(String key) {
        return Double.parseDouble(getProperty(key));
    }    
    
    public Double[] getDoubleArray(String key) {
        return (Double[])get(key);
    }       

    /**
     * Get specified property as boolean. .
     * @param key name of the property
     * @return 
     */      
    public boolean getBoolean(String key) {
        return Boolean.parseBoolean(getProperty(key));
    }    
    
    /**
     * Get specified property as String .
     * @param key name of the property
     * @return 
     */      
    public String getString(String key) {
        return getProperty(key);
    }    
}
