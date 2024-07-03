/*
 * ARX Data Anonymization Tool
 * Copyright 2012 - 2023 Fabian Prasser and contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.deidentifier.arx.gui.view.impl.wizard;

import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.DataType;
import org.deidentifier.arx.aggregates.HierarchyBuilder;
import org.deidentifier.arx.aggregates.HierarchyBuilderCustomBased;
import org.deidentifier.arx.framework.data.DataMatrix;
import org.deidentifier.arx.gui.Controller;
import org.deidentifier.arx.gui.model.Model;
import org.deidentifier.arx.gui.resources.Resources;

import java.util.ArrayList;
import java.util.List;

public class HierarchyWizardModelCustom <T> extends HierarchyWizardModelAbstract<T>{
    /**
     * Creates a new instance. Here we need to implement the code that reads python files
     *
     */
    private String path;
    private String[] data;
    private String[] allData;
    private List<String> parameters;
    private String errorMessage;
    private boolean mayBuild;
    private HierarchyBuilderCustomBased<T> builder;
    private Controller controller;
    private boolean useAllData;

    private String delimiter;

    public HierarchyWizardModelCustom(DataType<T> dataType, String[] data, String path, Controller c) {
        super(data);
        parameters = new ArrayList<>();
        errorMessage = "";
        this.data = data;
        this.path = path;
        mayBuild = true;
        builder = null;
        controller = c;
        useAllData = false;
        delimiter = ",";

        // Update
        this.update();
    }
    public void setMayBuild(boolean b){
        mayBuild = b;
    }
    public void setDelimiter(String s){
        delimiter = s;
    }
    public void setAllData(){
        Model model = controller.getModel();
        String attr = model.getSelectedAttribute();
        int column = model.getInputConfig()
                .getInput()
                .getHandle()
                .getColumnIndexOf(attr);
        int rows = controller.getModel().getInputConfig().getInput().getHandle().getNumRows();
        this.allData = new String[rows];
        for (int i = 0; i < rows; i++){
            allData[i] = model.getInputConfig().getInput().getHandle().getValue(i, column);
        }
    }
    public void setUseAllData(boolean all){
        useAllData = all;
    }

    @Override
    public HierarchyBuilderCustomBased<T> getBuilder(boolean serializable) {

        List<String[]> modelData = new ArrayList<>();
        if (useAllData){
            modelData.add(allData);
        }
        else{
            modelData.add(data);
        }
        HierarchyBuilderCustomBased<T> builder = HierarchyBuilderCustomBased.create(path,modelData,parameters,delimiter);
        this.builder = builder;
        return builder;
    }

    @Override
    public void parse(HierarchyBuilder<T> hierarchyBuilder) {
        if (!(hierarchyBuilder instanceof HierarchyBuilderCustomBased)) {
            return;
        }
        HierarchyBuilderCustomBased<T> builder = (HierarchyBuilderCustomBased<T>)hierarchyBuilder;

        update();
    }

    @Override
    public void updateUI(HierarchyWizard.HierarchyWizardView sender) {
        //Empty design
    }

    /**
     * Update the model and all UI components.
     */
    @Override
    public void update(){
        super.update();
        updateUI(null);
    }

    @Override
    protected void build() {
        super.hierarchy = null;
        super.error = null;
        super.groupsizes = null;
        this.errorMessage = "";
        if (data==null) return;

        if (mayBuild){
            HierarchyBuilderCustomBased<T> builder = getBuilder(false);

            try {
                AttributeType.Hierarchy hierarchy = builder.build(true);
                String[][] h = hierarchy.getHierarchy();
                int lengthData = data.length;
                if (useAllData){
                    lengthData = allData.length;
                }
                if (hierarchy.getHierarchy().length < lengthData){
                    for (int i =0; i < h.length; i++){
                        errorMessage += h[i][0];
                    }
                }
                super.groupsizes = builder.getGroupSizes();
                super.hierarchy = hierarchy;
                this.builder = builder;
            } catch(Exception e){
                super.error = Resources.getMessage("HierarchyWizardModelRedaction.1"); //$NON-NLS-1$
            }
        }
        else{
           AttributeType.Hierarchy hierarchy = builder.getHierarchy();
            if (hierarchy != null){
                String[][] h = hierarchy.getHierarchy();
                int dataLength = data.length;
                if (useAllData){
                    dataLength = allData.length;
                }
                if (hierarchy.getHierarchy().length < dataLength){
                    System.out.println();
                    for (int i =0; i < h.length; i++){
                        errorMessage += h[i][0];
                    }
                }
                super.groupsizes = builder.getGroupSizes();
                super.hierarchy = hierarchy;
            }
        }

    }
    public String getErrorMessage(){
        return this.errorMessage;
    }
    public void setPath(String path) {
        this.path = path;
    }
    public void setParameters(List<String> parameters){
        this.parameters = parameters;
    }
}
