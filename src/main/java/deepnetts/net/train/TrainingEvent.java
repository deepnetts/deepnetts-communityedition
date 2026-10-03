package deepnetts.net.train;

/**
 * TrainingEvent is used to notify interested parties that training event has happened.
 * It contains a type and a source of the event.
 */
public final class TrainingEvent {
    /* Add generic T extends Trainer instead of BackpropagationTrainer */
    
    private final BackpropagationTrainer source;
    private final Type type;

    /**
     * Type of a training event.
     */
    public static enum Type {
        /**
         * Specifies a training started event
         */        
        STARTED, 
        
        /**
         * Specifies a training stopped event
         */        
        STOPPED,
        
        /**
         * Specifies a training epoch started event
         */                
        EPOCH_STARTED,        
        
        /**
         * Specifies a training epoch finished event
         */        
        EPOCH_FINISHED,
        
        /**
         * Specifies a mini batch (batch of items) finished event
         */           
        MINI_BATCH,
        
        /**
         * Specifies a training iteration for (single data item) finished event
         */           
        ITERATION_FINISHED;
    }

    /**
     * Constructs a new TrainingEvent with specified source and type.
     * @param source source of the event, typically some training algorithm
     * @param type type of the event
     */
    public TrainingEvent(final BackpropagationTrainer source, final Type type) {
        this.source = source;
        this.type = type;
    }

    /**
     * Gets the source of the event.
     * Usually the source of the event is some training algorithm.
     * @return the source of the event.
     */
    public BackpropagationTrainer getSource() {
        return source;
    }

    /**
     * Gets the type of the event.
     * @return the type of the event
     */
    public Type getType() {
        return type;
    }

    /**
     * Specifies that training has started.
     */     
    public static Type STARTED = Type.STARTED;
    
    /**
     * Specifies that training has stopped.
     */        
    public static Type STOPPED = Type.STOPPED;
    
    /**
     * Specifies that training epoch has started.
     */
    public static Type EPOCH_STARTED = Type.EPOCH_STARTED;    
    
    /**
     * Specifies that training epoch has finished.
     */
    public static Type EPOCH_FINISHED = Type.EPOCH_FINISHED;
   
    /**
     * Specifies mini batch event.
     */    
    public static Type MINI_BATCH = Type.MINI_BATCH;
    
    /**
     * Specifies that training iteration has finished.
     */
    public static Type ITERATION_FINISHED = Type.ITERATION_FINISHED;

}