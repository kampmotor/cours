package com.cours.helper;

/**
 * Example usage of the JavaCodeHelper utility class.
 */
public class ExampleUsage {

    public static void main(String[] args) {
        JavaCodeHelper helper = new JavaCodeHelper();
        
        System.out.println("=== Java Code Helper Examples ===\n");
        
        // Example 1: Validate identifiers
        System.out.println("1. Identifier Validation:");
        System.out.println("   'myVariable' is valid: " + helper.isValidIdentifier("myVariable"));
        System.out.println("   '123invalid' is valid: " + helper.isValidIdentifier("123invalid"));
        System.out.println("   'class' is valid: " + helper.isValidIdentifier("class"));
        System.out.println();
        
        // Example 2: Naming conventions
        System.out.println("2. Naming Conventions:");
        System.out.println("   'my_variable_name' to camelCase: " + helper.toCamelCase("my_variable_name"));
        System.out.println("   'my-class-name' to PascalCase: " + helper.toPascalCase("my-class-name"));
        System.out.println();
        
        // Example 3: Generate getter/setter names
        System.out.println("3. Getter/Setter Generation:");
        System.out.println("   Field 'username' -> Getter: " + helper.generateGetterName("username"));
        System.out.println("   Field 'password' -> Setter: " + helper.generateSetterName("password"));
        System.out.println();
        
        // Example 4: Code analysis
        String sampleCode = """
            public class Calculator {
                public int add(int a, int b) {
                    return a + b;
                }
                
                private int subtract(int a, int b) {
                    return a - b;
                }
            }
            """;
        
        System.out.println("4. Code Analysis:");
        System.out.println("   Sample code has " + helper.countLines(sampleCode) + " lines");
        System.out.println("   Methods found: " + helper.extractMethodNames(sampleCode));
        System.out.println();
        
        // Example 5: Code formatting
        System.out.println("5. Code Formatting:");
        String codeWithComment = "int x = 5; // initialize x";
        System.out.println("   Before: " + codeWithComment);
        System.out.println("   After removing comments: " + helper.removeSingleLineComments(codeWithComment));
        System.out.println();
        
        String unindentedCode = "public void test() {\n    System.out.println(\"Hello\");\n}";
        System.out.println("   Indented code (4 spaces):");
        System.out.println(helper.indentCode(unindentedCode, 4));
    }
}
