package deepnetts.util;

import java.util.concurrent.Callable;
import java.util.function.IntConsumer;

/**
 * Callable int consumer. 
 * Applies given IntConsumer to specified int range.
 * 
 */
public final class CallableIntConsumer implements Callable<Void> {
        private final int from, to;
        private final IntConsumer func;
        
        public CallableIntConsumer(int from, int to, final IntConsumer func) {
            this.from = from;
            this.to = to;
            this.func = func;
        }        

        @Override
        public Void call() throws Exception {
            for (int cellIdx = from; cellIdx < to; cellIdx++) {
                func.accept(cellIdx);
//                if (Thread.interrupted()) {
//                    throw new InterruptedException();
//                }                
            }
                        
            return null;
        }    
}
