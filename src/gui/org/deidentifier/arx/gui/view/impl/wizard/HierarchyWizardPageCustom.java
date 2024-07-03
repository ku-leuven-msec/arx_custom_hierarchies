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

import org.deidentifier.arx.gui.Controller;
import org.deidentifier.arx.gui.resources.Resources;
import org.deidentifier.arx.gui.view.SWTUtil;
import org.deidentifier.arx.io.PythonFileReader;
import org.eclipse.swt.SWT;
import org.eclipse.swt.events.SelectionAdapter;
import org.eclipse.swt.events.SelectionEvent;
import org.eclipse.swt.graphics.Color;
import org.eclipse.swt.graphics.Cursor;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.eclipse.swt.widgets.Button;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Label;

import java.util.ArrayList;
import java.util.List;

/**
 * A page for configuring the custom builder.
 */
public class HierarchyWizardPageCustom<T> extends HierarchyWizardPageBuilder<T> {
    private Button btnChoose;
    private Button btnApplyHierarchy;
    private Button btnGetParameters;
    private Combo comboLocation;
    private Label lblLocation;

    private final HierarchyWizardModelCustom<T> model;
    private final Controller controller;
    private Group group;
    private Cursor c;
    private Text errorText;
    private Display display;
    private Button useDistinctOnly;
    private Label lblDelimiter;
    private Text txtDelimiter;
    private Label txtDescription;


    /**
     * Creates a new instance.
     *
     * @param wizard
     * @param model
     * @param finalPage
     */
    public HierarchyWizardPageCustom(final HierarchyWizard<T> wizard,final Controller controller,
                                        final HierarchyWizardModel<T> model,
                                        final HierarchyWizardPageFinal<T> finalPage) {
        super(wizard, model.getCustomModel(), finalPage);
        this.model = model.getCustomModel();
        this.controller = controller;
        c = null;
        setTitle("Custom hierarchy page"); //$NON-NLS-1$
        setDescription("Select the Python file needed to create the required hierarchy"); //$NON-NLS-1$
        setPageComplete(true);
    }

    @Override
    public void createControl(final Composite parent) {

        Composite composite = new Composite(parent, SWT.NONE);
        display = composite.getDisplay();
        composite.setLayout(new GridLayout(4, false));

        lblLocation = new Label(composite, SWT.NONE);
        lblLocation.setLayoutData(new GridData(SWT.LEFT, SWT.CENTER, false, false, 1, 1));
        lblLocation.setText(Resources.getMessage("ImportWizardPageCSV.7")); //$NON-NLS-

        comboLocation = new Combo(composite, SWT.READ_ONLY);
        comboLocation.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false, 2, 1));

        btnChoose = new Button(composite, SWT.NONE);
        btnChoose.setText(Resources.getMessage("ImportWizardPageCSV.8"));
        btnChoose.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, false, false, 1, 1));//$NON-NLS-1$
        btnChoose.addSelectionListener(new SelectionAdapter() {

            /**
             * Opens a file selection dialog for exe files
             *
             * If a valid exe file was selected, it is added to
             * {@link #comboLocation} when it wasn't already there. It is then
             * preselected within {@link #comboLocation} and the page is
             * evaluated {@see #evaluatePage}.
             */
            @Override
            public void widgetSelected(SelectionEvent arg0) {

                /* Open file dialog */
                final String path = controller.actionShowOpenFileDialog(getShell(),
                        "*.exe"); //$NON-NLS-1$
                if (path == null) {
                    return;
                }

                /* Check whether path was already added */
                if (comboLocation.indexOf(path) == -1) {
                    comboLocation.add(path, 0);
                }

                /* Select path and notify comboLocation about change */
                comboLocation.select(comboLocation.indexOf(path));
                comboLocation.notifyListeners(SWT.Selection, null);
            }
        });

        lblDelimiter = new Label(composite, SWT.NONE);
        lblDelimiter.setLayoutData(new GridData(SWT.LEFT, SWT.CENTER, false, false, 2, 1));
        lblDelimiter.setText("Delimiter to split hierarchy values");

        txtDelimiter = new Text(composite, SWT.WRAP);
        txtDelimiter.setLayoutData(new GridData(SWT.LEFT, SWT.CENTER, false, false, 2, 1));
        txtDelimiter.setText(",");
        txtDelimiter.setEditable(true);

        group = new Group(composite, SWT.SHADOW_ETCHED_IN);
        group.setText("Parameters");
        group.setLayoutData(new GridData(SWT.LEFT, SWT.CENTER, true, false, 3, 3));
        group.setLayout(SWTUtil.createGridLayout(3, false));

        useDistinctOnly = new Button(composite,SWT.CHECK);
        useDistinctOnly.setText("Only distinct values");
        useDistinctOnly.setLayoutData(new GridData(SWT.LEFT, SWT.CENTER, false, false, 1, 1) );
        useDistinctOnly.setSelection(true);

        btnApplyHierarchy = new Button(composite, SWT.NONE);
        btnApplyHierarchy.setText("Apply hierarchy");
        btnApplyHierarchy.setLayoutData(new GridData(SWT.LEFT, SWT.CENTER, false, false, 1, 1) );
        btnApplyHierarchy.addSelectionListener(new SelectionAdapter() {

            @Override
            public void widgetSelected(SelectionEvent arg0) {
                model.setMayBuild(true);
                Display d = composite.getDisplay();
                Cursor cursor = new Cursor(d,SWT.CURSOR_WAIT);
                Shell s = composite.getShell();
                s.setCursor(cursor);

                List<String> extraParameters = new ArrayList<>();
                Control[] c = group.getChildren();
                for (int i = 0; i < c.length; i++){
                    Text t = (Text) c[i];
                    if (t.getEditable() && t.isEnabled()){
                        extraParameters.add(t.getText());
                    }
                }
                makeHierarchy(comboLocation.getItem(comboLocation.getSelectionIndex()), extraParameters);
                cursor = new Cursor(d, SWT.CURSOR_ARROW);
                s.setCursor(cursor);
            }
        });
        btnGetParameters = new Button(composite, SWT.NONE);
        btnGetParameters.setText("Get script parameters");
        btnGetParameters.setLayoutData(new GridData(SWT.LEFT, SWT.CENTER, false, false, 1, 1) );
        btnGetParameters.addSelectionListener(new SelectionAdapter() {

            @Override
            public void widgetSelected(SelectionEvent arg0) {
                model.setMayBuild(true);
                Display d = composite.getDisplay();
                Color red = d.getSystemColor(SWT.COLOR_RED);
                errorText.setForeground(red);
                Cursor c = new Cursor(d,SWT.CURSOR_WAIT);
                Shell s = composite.getShell();
                s.setCursor(c);
                removePreviousParameters();
                getParameters(comboLocation.getItem(comboLocation.getSelectionIndex()));
                c = new Cursor(d, SWT.CURSOR_ARROW);
                s.setCursor(c);
            }
        });
        txtDescription = new Label(composite,SWT.WRAP);
        txtDescription.setLayoutData(new GridData(SWT.FILL, SWT.BOTTOM, true, true, 4, 4) );

        errorText = new Text(composite,SWT.WRAP);
        errorText.setEditable(false);
        errorText.setLayoutData(new GridData(SWT.FILL, SWT.BOTTOM, true, true, 4, 2) );

        updatePage();
        setControl(composite);
    }

    public void removePreviousParameters(){
        Control[] c = group.getChildren();

        for (Control child: c){
            child.dispose();
        }
        group.update();
        group.layout();
        updatePage();
    }
    public void getParameters(String path){

        PythonFileReader pythonFileReader = new PythonFileReader(path);
        List<String[]> parameters = pythonFileReader.getParameters(path);
        for (int i = 0; i < parameters.size() -1; i++) {
            String[] s = parameters.get(i);
            Text t = new Text(group, SWT.NONE);
            t.setText(s[0] + " (" + s[1] + ")");
            t.setEnabled(true);
            t.setEditable(false);
            Text input = new Text(group, SWT.NONE);
            input.setText(s[2]);
            input.setEditable(true);
            input.setEnabled(true);

            Text description = new Text(group, SWT.NONE);
            description.setText(s[3]);
            description.setEditable(false);
            description.setEnabled(true);
            description.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
        }
        txtDescription.setText(parameters.get(parameters.size() -1)[0]);

        group.update();
        group.layout();
        group.pack();
        group.getParent().layout();

        model.setVisible(true);
        model.setMayBuild(false);
    }

    public void makeHierarchy(String path, List<String> parameters){
        if (!useDistinctOnly.getSelection()){
            model.setAllData();
        }
        model.setDelimiter(txtDelimiter.getText());
        model.setUseAllData(!useDistinctOnly.getSelection());

        model.setPath(path);
        model.setParameters(parameters);
        model.build();
        if (model.getErrorMessage() != ""){
            //Set message
            errorText.setText(model.getErrorMessage());
            errorText.setVisible(true);
        }
        else{
            errorText.setText("");
            errorText.setVisible(false);
        }
        model.setVisible(true);
        model.setMayBuild(false);
        //model.update();
    }

    @Override
    public void updatePage() {
        model.update();
    }
}
