package org.deidentifier.arx.io;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class PythonFileReader {
    private String delimiter;
    private final String path;
    private List<Set<String>> groups;
    private List<String[]> data;
    private List<String> params;

    public PythonFileReader(String path, String delimiter, List<String[]> inputData, List<String> parameters){
        this.delimiter = delimiter;
        this.path = path;
        this.groups = new ArrayList<>();
        data = inputData;
        params = new ArrayList<>();
        if (parameters != null){
            if (!parameters.isEmpty()){
                params.addAll(parameters);
            }
        }

    }

    public PythonFileReader(String path){
        this.path = path;
    }


    public List<String[]> getParameters(String path){
        ProcessBuilder buider = new ProcessBuilder(path);
        Process p;
        try {
            p = buider.start();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        PrintWriter writer = new PrintWriter(new OutputStreamWriter(p.getOutputStream()));

        // first let the python script know how many elements the dataset has, this could also be done as script argument
        writer.println(1);
        // write our data one line at a time
        writer.println("getParameters");
        writer.close();

        // read all the output our python returns
        BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
        List<String[]> params = new ArrayList<>();
        String line;
        try{
            while ((line = reader.readLine()) != null) {
                params.add(splitParamArray(line));
            }
        }catch (IOException e) {
            throw new RuntimeException(e);
        }
        return params;
    }
    public String[] splitParamArray(String param){
        String[] split = param.split("'");
        if (split.length > 1){
            String[] result = new String[split.length/2];
            int counter = 0;
            for (int i = 1; i < split.length;i+=2){
                result[counter] = split[i];
                counter++;
            }
            return result;
        }
        return split;
    }

    public List<String[]> hierarchyDataLines(){
        ProcessBuilder buider = new ProcessBuilder(path);
        Process p;
        try {
            p = buider.start();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        PrintWriter writer = new PrintWriter(new OutputStreamWriter(p.getOutputStream()));
        writer.println(params.size() + data.get(0).length);
        for (String s : params){
            writer.println(s);
        }
        for (String s : data.get(0)){
            writer.println(s);
        }
        writer.close();

        List<String> outputLines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                 outputLines.add(line);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return splitAllLinesByDelimiter(outputLines);
    }

    public void getGroups(List<String[]> items){
        this.groups = new ArrayList<>();
        for (int j = 0; j < items.get(0).length;j++){
            this.groups.add(new HashSet<>());
        }
        for (int i =0 ; i < items.size(); i++){
            for (int k = 0; k < items.get(i).length; k++){
                this.groups.get(k).add(items.get(i)[k]);
            }
        }
    }
    public int[] getGroups(){
        int[] groupSizes = new int[this.groups.size()];
        for (int i = 0; i < groupSizes.length; i++){
            groupSizes[i] = groups.get(i).size();
        }
        return groupSizes;
    }

    public List<String[]> splitAllLinesByDelimiter(List<String> results){
        List<String[]> splitStrings = new ArrayList<>();
        for (String s : results) {
            splitStrings.add(s.split(delimiter));
        }
        getGroups(splitStrings);
        return splitStrings;
    }
}
