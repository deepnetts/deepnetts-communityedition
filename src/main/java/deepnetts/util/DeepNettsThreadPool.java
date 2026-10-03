package deepnetts.util;

import deepnetts.core.DeepNetts;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Dedicated thread pool for Deep Netts Engine.
 * 

 */
public class DeepNettsThreadPool {

    private ExecutorService es;
    private int threadNum;
    private int maxThreadNum;
    
    public DeepNettsThreadPool() {
        maxThreadNum = DeepNetts.getInstance().getMaxThreads();//getMaxThreadsNum();
        threadNum = maxThreadNum;
        es = Executors.newFixedThreadPool(threadNum);
    }
    
    public DeepNettsThreadPool(int threadNum) {
        this.threadNum = threadNum;
        es = Executors.newFixedThreadPool(threadNum);
    }    
    
    
    // posto ovaj radi sa invokeAll, moze da se koristi umesto one zavrzaleme sa cyclic barriers verovatno - istestiraj
    public void run(Collection<Callable<Void>> tasks) throws InterruptedException {
         es.invokeAll(tasks);
    }
           
    /**
     * Submit a single task to thread pool.
     * @param task
     * @return 
     */
    public Future<?> submit(Callable<?> task) {
        return es.submit(task);
    }

    public void run(Runnable task) {      
        es.submit(task);
    }
    
    // maintain a list of running trainings  and allow shitdown only when trainings are unsubscribed
    public void shutdown() {
        es.shutdown();
    }
    
    public void shutdownNow() {
        es.shutdownNow();
    }    

    final public int getThreadNum() {
        return threadNum;
    }
    
    public void setThreadNum(int threadNum) {
        this.threadNum = threadNum;
        es = Executors.newFixedThreadPool(threadNum); // ne bih smeo ovde da menjam jer sam ih sve inicijalizovao
    }
    
    public static int getMaxThreadsNum() {

        int threads = Runtime.getRuntime().availableProcessors();

        threads /= 2; // 2 logical processors for 1 core in most hardware architectures
        threads--; // One thread for the OS

        if (threads < 1) {
            threads = 1;
        }

        return threads;
    }       
    
    // make this non static since we allready know num of threads
    public static int[] calculateCellsPerThread(int width, int threadNum) {
        int[] threads = new int[threadNum];
        int cpt = width / threadNum;
        
        for(int i=0; i<threadNum; i++) {
            threads[i] = cpt;
        }
                        
        if (width % threadNum !=0) {
            int rest = width % threadNum;
            
            for(int i=0; i< rest; i++) {
                threads[i] = threads[i] + 1;
            }
        }
        
        return threads;
    }      

    public ExecutorService getExecutorService() {
        return es;
    }
    
    

    
 }