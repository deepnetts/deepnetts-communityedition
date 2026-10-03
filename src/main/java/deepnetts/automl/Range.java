package deepnetts.automl;

/**
 * A value range of type T for the parameter.
 * Mainly used for numeric parameters in order to specify min, max, step or search strategy.
 *
 * @param <T> Type of values.
 */
public final class Range<T> {
    private final T min, max;
    private T step;
    private int randomNum; // if used for random sampling from given range
    private String strategy; // strategy to select specific values from the given range

    public Range(T min, T max) {
        this.min = min;
        this.max = max;
    }
    
    public Range(T min, T max, T step) {
        this.min = min;
        this.max = max;
        this.step = step;
    }    

    public T getMin() {
        return min;
    }

    public T getMax() {
        return max;
    }

    public T getStep() {
        return step;
    }
       
    public static <T> Range<T> of(T min, T max) {
        return new Range(min, max);
    }    
    
    public static <T> Range<T> of(T min, T max, T step) {
        return new Range(min, max).step(step);
    }      
    
    public Range<T> randomNum(int randomNum) {    
        this.randomNum = randomNum;
        return this;
    }
    
    public Range<T> step(T step) {    
        this.step = step;
        return this;
    }    

    public int getRandomNum() {
        return randomNum;
    }
    
    public Range<T> strategy(String strategy) {
        this.strategy = strategy;
        return this;        
    }

    public String getStrategy() {
        return strategy;
    }
    
    
       
    
}
