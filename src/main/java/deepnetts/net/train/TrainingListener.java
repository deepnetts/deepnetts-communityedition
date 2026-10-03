package deepnetts.net.train;
import java.util.EventListener;

/**
 * The listener interface for receiving notifications about training events.
 */
public interface TrainingListener extends EventListener {
    
    /**
     * Invoked when a training event occurs.
     * @param event the training event
     */
    public void handleEvent(TrainingEvent event);
}
