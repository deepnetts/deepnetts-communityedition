package deepnetts.net.loss;

/**
 * Supported types of Loss Functions in Deep Netts engine.
 * Currently supported loss functions are Mean Squared Error (commonly used for regression) and Cross Entropy (commonly used for classification)
 *
 * @see  LossFunction
 */
public enum LossType {
    /**
     * Mean Squared Error loss, used for regression tasks, implemented by {@link MeanSquaredErrorLoss}
     */
    MEAN_SQUARED_ERROR("MEAN_SQUARED_ERROR"),

    /**
     * Cross Entropy Loss, used for classificaton tasks, implemented by {@link CrossEntropyLoss}
     */
    CROSS_ENTROPY("CROSS_ENTROPY");

    private final String name;

    private LossType(String name) {
        this.name = name;
    }

    public boolean equalsName(String otherName) {
        return name.equals(otherName);
    }

    public static LossType of(Class lossClass) {
        if (lossClass.equals(MeanSquaredErrorLoss.class)) {
            return MEAN_SQUARED_ERROR;
        } else if (lossClass.equals(CrossEntropyLoss.class) || lossClass.equals(BinaryCrossEntropyLoss.class)) {
            return CROSS_ENTROPY;
        }

       throw new RuntimeException("Unknown loss type!");
    }

    @Override
    public String toString() {
       return this.name;
    }
}