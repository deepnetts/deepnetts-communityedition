package deepnetts.data;

/**
 * Data pre-processing abstraction.
 */
public interface Preprocessing<T> {
        public void apply(T input);
}
