/**  
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation 
 *  based learning and image recognition.
 * 
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */
    
package deepnetts.util;

import java.util.HashMap;

/**
 * TODO: setMethod, throw meannigfull exceptions
 * 

 */
public class Parameters {
    HashMap<String, Parameter> parameters;

    public Parameters() {
        parameters = new HashMap<>();
    }
    
    public Parameter<?> get(String name) {
        return parameters.get(name);
    }
    
    public Float getFloat(String name) {
        return  (Float) parameters.get(name).getValue();
    }    
    
    public Integer getInteger(String name) {
        return  (Integer) parameters.get(name).getValue();
    }    
    
    public Boolean getBoolean(String name) {
        return (Boolean) parameters.get(name).getValue();
    }
    
    public String getString(String name) {
        return (String) parameters.get(name).getValue();
    }    
       
    public void put(Parameter<?> param){
        parameters.put(param.getName(), param);
    }
    
    
    
}