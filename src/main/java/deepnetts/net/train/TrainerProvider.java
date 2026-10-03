package deepnetts.net.train;

/**
 * This interface is implemented by trainable deep learning models,
 * in order to provide access to training algorithm.
 */
public interface TrainerProvider<T extends Trainer> {
        public T getTrainer();
        public void setTrainer(T trainer);
}
