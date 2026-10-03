package deepnetts.accl;

import java.lang.foreign.MemorySegment;

public interface AcceleratorTensorBridge extends AutoCloseable  {
    public void copyToDev(); // copyTo
    public void copyToHost(); // copyFrom    

    public boolean isAllocatedOnDev();

    public MemorySegment allocateAndCopyToDev();
}
