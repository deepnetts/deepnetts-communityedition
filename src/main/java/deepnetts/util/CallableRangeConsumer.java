package deepnetts.util;

import java.util.concurrent.Callable;

/**
 * Callable range consumer. 
 * Calls given RangeConsumer function with specified int (from, to) range.
 * Has CyclicBarrier for synchronization.
 * 
 * @author Zoran Sevarac
 */
public final class CallableRangeConsumer implements Callable<Void> {
        private final int from, to;
        private final RangeConsumer func;
        
        public CallableRangeConsumer(int from, int to, final RangeConsumer func) {
            this.from = from;
            this.to = to;
            this.func = func;
        }        

        @Override
        public Void call() throws Exception {            
            func.accept(from, to);
            return null;
        }    
}
