package deepnetts.util;

import deepnetts.core.DeepNetts;
import static deepnetts.core.DeepNetts.LOGGER;
import java.io.IOException;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

public class DeepNettsLogManager {

    public static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());   
    private FileHandler fileHandler;
    
    private static final Formatter formatter = new Formatter() {
                @Override
                public String format(LogRecord record) {
                    StringBuilder sb= new StringBuilder();
                    sb.append(record.getMessage()).append(System.lineSeparator());
                    return sb.toString();
                }
            };    
    
    static {
        LOGGER.setUseParentHandlers(false);
        ConsoleHandler consoleHandler = new ConsoleHandler();
        consoleHandler.setFormatter(formatter);
        
        LOGGER.addHandler(consoleHandler);
        //DeepNettsLogManager logManager= new DeepNettsLogManager();
        //logManager.setFileHandler("logs/deepnetts_training.log");
                
//        Handler[] handlers = LOGGER.getHandlers(); // zasto nema handlera?
//        for(Handler hnd: handlers) {
//            hnd.setFormatter(formatter);
//        }
    }    
    
    public FileHandler setFileHandler(String fileName) {
        try {
            fileHandler = new FileHandler(fileName);
            fileHandler.setFormatter(formatter);
            LOGGER.addHandler(fileHandler);
            return fileHandler;
        } catch (IOException ex) {
            Logger.getLogger(DeepNettsLogManager.class.getName()).log(Level.SEVERE, null, ex);
        } catch (SecurityException ex) {
            Logger.getLogger(DeepNettsLogManager.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    
    public void removeHandler(Handler handler) {
        LOGGER.removeHandler(handler);        
    }
    
    public FileHandler getFileHandler() {
        return fileHandler;
    }
    
    // kako zatvortiti handler - flush, close na get fileHandler
    
}
