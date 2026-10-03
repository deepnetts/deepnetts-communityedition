package deepnetts.automl;

import java.util.List;
import java.util.Objects;

/**
 * A single parameter with name and all possible values to try.
 * 
 * @param <T> Type of parameter values
 */
public final class Parameter<T> {
    
    /**
     * Name of this parameter
     */
    private final String name;
    
    /**
     * List of all possible values for this parameter
     */
    private final List<T> values; 
    // range of parameters is always pre generated as listof possible values

    
    public Parameter(String name, List<T> values) {
        this.name = name;
        this.values = values;
    }
    
 
    public String getName() {
        return name; 
    }
    
    /**
     * Returns the list of possible values for this parameter
     * 
     * @return 
     */
    public List<T> getValues() {
        return values;
    }
    
    /**
     * If it does not hold list but a single value
     * @return 
     */
    public T getValue() {
        return values.get(0);
    }
        
    @Override
    public int hashCode() {
        int hash = 7;
        hash = 73 * hash + Objects.hashCode(this.name);
        hash = 73 * hash + Objects.hashCode(this.values);
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Parameter<?> other = (Parameter<?>) obj;
        if (!Objects.equals(this.name, other.name)) {
            return false;
        }
        if (!Objects.equals(this.values, other.values)) {
            return false;
        }
        return true;
    }

    @Override
    public String toString() {
        return name + ":" + values.get(0);
    }    
    
}