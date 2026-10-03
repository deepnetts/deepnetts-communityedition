package deepnetts.util;

import java.util.Arrays;

/**
 * CSV file format options: delimiter, header(column names) and column types. 
 */
public class CsvFormat {
    private String delimiter;
    private boolean hasHeader;
    private String[] columnNames;
    private ColumnType[] columnTypes;
    private int numColumns;
    
    public String getDelimiter() {
        return delimiter;
    }

    public void setDelimiter(String delimiter) {
        this.delimiter = delimiter;
    }

    public boolean isHasHeader() {
        return hasHeader;
    }

    public void setHasHeader(boolean hasHeader) {
        this.hasHeader = hasHeader;
    }

    public String[] getColumnNames() {
        return columnNames;
    }

    public void setColumnNames(String[] columnNames) {
        this.columnNames = columnNames;
        this.numColumns = columnNames.length;
    }

    public ColumnType[] getColumnTypes() {
        return columnTypes;
    }

    public void setColumnTypes(ColumnType[] columnTypes) {
        this.columnTypes = columnTypes;
    }

    public int getNumColumns() {
        return numColumns;
    }
        
    @Override
    public String toString() {
        return "CsvFormat{" + "delimiter=" + delimiter + ", hasHeader=" + hasHeader + ", columnNames=" + Arrays.toString(columnNames) + ", columnTypes=" + Arrays.toString(columnTypes) + '}';
    }
    
}