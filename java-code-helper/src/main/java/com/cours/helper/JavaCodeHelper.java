package com.cours.helper;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Java Code Helper - A utility class to assist with common Java programming tasks.
 * This helper provides methods for code analysis, formatting, and validation.
 */
public class JavaCodeHelper {

    /**
     * Validates if a given string is a valid Java identifier.
     * 
     * @param identifier the string to check
     * @return true if the identifier is valid, false otherwise
     */
    public boolean isValidIdentifier(String identifier) {
        if (identifier == null || identifier.isEmpty()) {
            return false;
        }
        
        if (!Character.isJavaIdentifierStart(identifier.charAt(0))) {
            return false;
        }
        
        for (int i = 1; i < identifier.length(); i++) {
            if (!Character.isJavaIdentifierPart(identifier.charAt(i))) {
                return false;
            }
        }
        
        return !isJavaKeyword(identifier);
    }

    /**
     * Checks if a string is a Java keyword.
     * 
     * @param word the word to check
     * @return true if it's a Java keyword, false otherwise
     */
    public boolean isJavaKeyword(String word) {
        String[] keywords = {
            "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
            "class", "const", "continue", "default", "do", "double", "else", "enum",
            "extends", "final", "finally", "float", "for", "goto", "if", "implements",
            "import", "instanceof", "int", "interface", "long", "native", "new", "package",
            "private", "protected", "public", "return", "short", "static", "strictfp",
            "super", "switch", "synchronized", "this", "throw", "throws", "transient",
            "try", "void", "volatile", "while", "true", "false", "null"
        };
        
        for (String keyword : keywords) {
            if (keyword.equals(word)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Converts a string to camelCase naming convention.
     * 
     * @param input the input string
     * @return the camelCase version of the string
     */
    public String toCamelCase(String input) {
        if (input == null || input.isEmpty()) {
            return input;
        }
        
        String[] words = input.split("[\\s_-]+");
        StringBuilder result = new StringBuilder(words[0].toLowerCase());
        
        for (int i = 1; i < words.length; i++) {
            if (!words[i].isEmpty()) {
                result.append(words[i].substring(0, 1).toUpperCase());
                if (words[i].length() > 1) {
                    result.append(words[i].substring(1).toLowerCase());
                }
            }
        }
        
        return result.toString();
    }

    /**
     * Converts a string to PascalCase naming convention.
     * 
     * @param input the input string
     * @return the PascalCase version of the string
     */
    public String toPascalCase(String input) {
        if (input == null || input.isEmpty()) {
            return input;
        }
        
        String[] words = input.split("[\\s_-]+");
        StringBuilder result = new StringBuilder();
        
        for (String word : words) {
            if (!word.isEmpty()) {
                result.append(word.substring(0, 1).toUpperCase());
                if (word.length() > 1) {
                    result.append(word.substring(1).toLowerCase());
                }
            }
        }
        
        return result.toString();
    }

    /**
     * Extracts all method names from a Java code snippet.
     * 
     * @param code the Java code to analyze
     * @return a list of method names found in the code
     */
    public List<String> extractMethodNames(String code) {
        List<String> methodNames = new ArrayList<>();
        
        // Pattern to match method declarations
        // Matches: [access modifiers] returnType methodName(parameters)
        Pattern pattern = Pattern.compile(
            "(public|private|protected)?\\s*(static)?\\s*[\\w<>\\[\\]]+\\s+([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\([^)]*\\)\\s*\\{"
        );
        
        Matcher matcher = pattern.matcher(code);
        while (matcher.find()) {
            String methodName = matcher.group(3);
            if (methodName != null && !methodName.isEmpty()) {
                methodNames.add(methodName);
            }
        }
        
        return methodNames;
    }

    /**
     * Counts the number of lines in a code snippet.
     * 
     * @param code the code to analyze
     * @return the number of lines
     */
    public int countLines(String code) {
        if (code == null || code.isEmpty()) {
            return 0;
        }
        return code.split("\n").length;
    }

    /**
     * Removes single-line comments from Java code.
     * 
     * @param code the Java code
     * @return the code without single-line comments
     */
    public String removeSingleLineComments(String code) {
        if (code == null) {
            return null;
        }
        return code.replaceAll("//.*", "");
    }

    /**
     * Indents code with the specified number of spaces.
     * 
     * @param code the code to indent
     * @param spaces the number of spaces for indentation
     * @return the indented code
     */
    public String indentCode(String code, int spaces) {
        if (code == null || code.isEmpty()) {
            return code;
        }
        
        String indentation = " ".repeat(spaces);
        String[] lines = code.split("\n");
        StringBuilder result = new StringBuilder();
        
        for (int i = 0; i < lines.length; i++) {
            result.append(indentation).append(lines[i]);
            if (i < lines.length - 1) {
                result.append("\n");
            }
        }
        
        return result.toString();
    }

    /**
     * Generates a getter method name from a field name.
     * 
     * @param fieldName the field name
     * @return the getter method name
     */
    public String generateGetterName(String fieldName) {
        if (fieldName == null || fieldName.isEmpty()) {
            return null;
        }
        return "get" + fieldName.substring(0, 1).toUpperCase() + fieldName.substring(1);
    }

    /**
     * Generates a setter method name from a field name.
     * 
     * @param fieldName the field name
     * @return the setter method name
     */
    public String generateSetterName(String fieldName) {
        if (fieldName == null || fieldName.isEmpty()) {
            return null;
        }
        return "set" + fieldName.substring(0, 1).toUpperCase() + fieldName.substring(1);
    }
}
