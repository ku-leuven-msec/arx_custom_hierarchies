package org.deidentifier.arx.aggregates;

import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.io.PythonFileReader;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class HierarchyBuilderCustomBased<T> extends HierarchyBuilder<T> implements Serializable {

    private String path;
    private List<String[]> modelData;
    private List<String> parameters;
    private int[] groupSizes;
    private AttributeType.Hierarchy hierarchy;
    private String delimiter;

    private HierarchyBuilderCustomBased(String path, List<String[]> modelData, List<String> parameters, String delimiter) {
        super(Type.CUSTOM_BASED);
        this.path = path;
        this.modelData = modelData;
        this.parameters = parameters;
        hierarchy = null;
        this.delimiter = delimiter;
    }
    public static <T> HierarchyBuilderCustomBased<T> create(String path, List<String[]> modelData, List<String> parameters, String delimiter){
        return new HierarchyBuilderCustomBased<T>(path, modelData, parameters, delimiter);
    }

    @Override
    public AttributeType.Hierarchy build() {
        return hierarchy;
    }
    public AttributeType.Hierarchy getHierarchy(){
        return this.hierarchy;
    }
    // Avoid that the gui keeps on rebuilding the hierarchy using the script while no changes where made.
    public AttributeType.Hierarchy build(boolean update) {
        if (path != null){
            PythonFileReader pythonFileReader = new PythonFileReader(path, String.valueOf(delimiter),modelData,parameters);
            List<String[]> hierarchyLines = pythonFileReader.hierarchyDataLines();

            this.groupSizes = pythonFileReader.getGroups();

            AttributeType.Hierarchy h = AttributeType.Hierarchy.create(hierarchyLines);
            this.hierarchy = h;
            return h;
        }
        else{
            AttributeType.Hierarchy h = AttributeType.Hierarchy.create();
            this.hierarchy = h;
            return h;
        }
    }
    //This needs to do the same as above.
    @Override
    public AttributeType.Hierarchy build(String[] data) {
        return build(true);
    }

    @Override
    public int[] prepare(String[] data) {
        return new int[0];
    }
    public int[] getGroupSizes(){
        return this.groupSizes;
    }
}
