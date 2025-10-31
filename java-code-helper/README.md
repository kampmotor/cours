# Java Code Helper

A utility library providing helpful methods for Java code operations, analysis, and formatting.

## Features

- **Identifier Validation**: Check if strings are valid Java identifiers
- **Keyword Detection**: Identify Java keywords
- **Naming Conventions**: Convert strings to camelCase and PascalCase
- **Code Analysis**: Extract method names from code snippets
- **Code Formatting**: Indent code, remove comments
- **Code Generation**: Generate getter/setter method names

## Building the Project

```bash
cd java-code-helper
mvn clean install
```

## Running Tests

```bash
mvn test
```

## Usage Examples

```java
import com.cours.helper.JavaCodeHelper;

public class Example {
    public static void main(String[] args) {
        JavaCodeHelper helper = new JavaCodeHelper();
        
        // Validate identifiers
        System.out.println(helper.isValidIdentifier("myVariable")); // true
        System.out.println(helper.isValidIdentifier("123invalid")); // false
        
        // Convert naming conventions
        System.out.println(helper.toCamelCase("my_variable_name")); // myVariableName
        System.out.println(helper.toPascalCase("my_class_name")); // MyClassName
        
        // Generate getter/setter names
        System.out.println(helper.generateGetterName("name")); // getName
        System.out.println(helper.generateSetterName("age")); // setAge
        
        // Check keywords
        System.out.println(helper.isJavaKeyword("class")); // true
    }
}
```

## Methods

### Validation
- `isValidIdentifier(String identifier)` - Validates Java identifiers
- `isJavaKeyword(String word)` - Checks if a word is a Java keyword

### Naming Conventions
- `toCamelCase(String input)` - Converts to camelCase
- `toPascalCase(String input)` - Converts to PascalCase

### Code Analysis
- `extractMethodNames(String code)` - Extracts method names from code
- `countLines(String code)` - Counts lines in code

### Code Formatting
- `removeSingleLineComments(String code)` - Removes // comments
- `indentCode(String code, int spaces)` - Indents code with specified spaces

### Code Generation
- `generateGetterName(String fieldName)` - Generates getter method name
- `generateSetterName(String fieldName)` - Generates setter method name

## Requirements

- Java 17 or higher
- Maven 3.6 or higher

## License

This is an educational project for course materials.
