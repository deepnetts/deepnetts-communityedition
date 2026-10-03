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

import java.util.Arrays;
import java.util.List;

/**
 *

 */
public class Parameter<T> {
    String name;
    T value;


    public Parameter(String name, T value) {
        this.value = value;
        this.name = name;
    }

    public String getName() {
        return name;
    }      
    
    public T getValue() {
        return value;
    }
    
    public static List<Parameter> listOf(Parameter ... a) {
        return Arrays.asList(a);
    }

    @Override
    public String toString() {
        return "Parameter{" + "name=" + name + ", value=" + value + '}';
    }

    
  
}
