package deepnetts.core;

import deepnetts.util.DeepNettsThreadPool;
//import deepnetts.license.LicenseChecker;
//import deepnetts.license.LicenseChecker.LicenceException;
import java.util.Properties;
import java.util.logging.ConsoleHandler;
import java.util.logging.Formatter;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Global configuration and settings for Deep Netts Engine.
 *
 *
 */
public final class DeepNetts {

    private static DeepNetts instance;

    public static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());

    // https://docs.oracle.com/javase/tutorial/essential/environment/sysprop.html
    private final Properties prop;
    private boolean useCuda = false;
    private boolean useVectorAPI = false;
    private boolean debugMode = false;
    private int maxThreads;

    static {
        LOGGER.setUseParentHandlers(false);
        LOGGER.addHandler(new ConsoleHandler());

        Handler[] handlers = LOGGER.getHandlers();
        for (Handler hnd : handlers) {
            hnd.setFormatter(new Formatter() {
                @Override
                public String format(LogRecord record) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(record.getMessage()).append(System.lineSeparator());
                    return sb.toString();
                }
            });
        }

    }

    private DeepNetts() {
        // load properties from META-INF file that should be generated during the build
        prop = new Properties();
        prop.put("version", "3.2.0");
        prop.put("vendor", "Deep Netts Technologies LLC");
        //  prop.put("multithreaded", "true");        
        prop.put("useCuda", "false");

        maxThreads = DeepNettsThreadPool.getMaxThreadsNum();
    }

    /**
     * Returns a singleton instance of a Deep Netts engine.The instance provides
     * global configuration settings for the Deep Netts runtime.
     *
     * @return global instance of a Deep Netts engine
     */
    public static DeepNetts getInstance() {
        if (instance == null) {
            instance = new DeepNetts();
        }

        return instance;
    }

    /**
     * Returns a specified property of the Deep Netts runtime environment.
     *
     * @param propName name of the property
     * @return property of the Deep Netts runtime
     */
    public String getProperty(String propName) {
        return prop.getProperty(propName);
    }

    /**
     * Validates the Deep Netts license.
     *
     * @throws LicenceException if license is not valid.
     */
//    public static void checkLicense() {
//        LicenseChecker checker = new LicenseChecker();
//        try {
//            checker.checkLicense();
//        } catch (NullPointerException npe) {
//            throw new LicenseChecker.LicenceException("Invalid license or license not found!");
//        }
//    }

    /**
     * Returns the Deep Netts version.
     *
     * @return
     */
    public String version() {
        return prop.getProperty("version");
    }

    /**
     * Returns all the configuration properties for Deep Netts.
     *
     * @return Deep Netts settings
     */
    public Properties getProperties() {
        return prop;
    }

    public boolean useCuda() {
        return useCuda;
    }

    public void setUseCuda(boolean useCuda) {
        this.useCuda = useCuda;
        prop.setProperty("useCuda", String.valueOf(useCuda));
    }

    public boolean useVectorAPI() {
        return useVectorAPI;
    }

    public void setUseVectorAPI(boolean useVectorAPI) {
        this.useVectorAPI = useVectorAPI;
        prop.setProperty("useVectorAPI", String.valueOf(useVectorAPI));
        setMaxThreads(1);// vector api is only for single threaded
    }

    public void setMaxThreads(int maxThreads) {
        if (maxThreads < 1) {
            throw new IllegalArgumentException("Max threads must be greater than zero!");
        }
        this.maxThreads = maxThreads;
        // getMaxThreadsNum
    }

    public int getMaxThreads() {
        return maxThreads;
    }

    public boolean isMultithreaded() {
        return maxThreads > 1;
    }

    public boolean isDebugMode() {
        return debugMode;
    }

    public void setDebugMode(boolean debugMode) {
        this.debugMode = debugMode;
    }
    
    
    
    

}
