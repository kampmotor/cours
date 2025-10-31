package com.cours.helper;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

/**
 * Test class for JavaCodeHelper.
 */
class JavaCodeHelperTest {

    private JavaCodeHelper helper;

    @BeforeEach
    void setUp() {
        helper = new JavaCodeHelper();
    }

    @Test
    void testIsValidIdentifier_ValidIdentifiers() {
        assertTrue(helper.isValidIdentifier("myVariable"));
        assertTrue(helper.isValidIdentifier("_privateVar"));
        assertTrue(helper.isValidIdentifier("$dollarVar"));
        assertTrue(helper.isValidIdentifier("variable123"));
    }

    @Test
    void testIsValidIdentifier_InvalidIdentifiers() {
        assertFalse(helper.isValidIdentifier("123invalid"));
        assertFalse(helper.isValidIdentifier("my-variable"));
        assertFalse(helper.isValidIdentifier("my variable"));
        assertFalse(helper.isValidIdentifier(""));
        assertFalse(helper.isValidIdentifier(null));
    }

    @Test
    void testIsValidIdentifier_JavaKeywords() {
        assertFalse(helper.isValidIdentifier("class"));
        assertFalse(helper.isValidIdentifier("public"));
        assertFalse(helper.isValidIdentifier("return"));
        assertFalse(helper.isValidIdentifier("if"));
    }

    @Test
    void testIsJavaKeyword() {
        assertTrue(helper.isJavaKeyword("class"));
        assertTrue(helper.isJavaKeyword("public"));
        assertTrue(helper.isJavaKeyword("static"));
        assertTrue(helper.isJavaKeyword("void"));
        assertFalse(helper.isJavaKeyword("myMethod"));
    }

    @Test
    void testToCamelCase() {
        assertEquals("myVariableName", helper.toCamelCase("my_variable_name"));
        assertEquals("myVariableName", helper.toCamelCase("my-variable-name"));
        assertEquals("myVariableName", helper.toCamelCase("my variable name"));
        assertEquals("test", helper.toCamelCase("test"));
        assertEquals("", helper.toCamelCase(""));
        assertEquals("", helper.toCamelCase("___"));
        assertEquals("", helper.toCamelCase("---"));
    }

    @Test
    void testToPascalCase() {
        assertEquals("MyClassName", helper.toPascalCase("my_class_name"));
        assertEquals("MyClassName", helper.toPascalCase("my-class-name"));
        assertEquals("MyClassName", helper.toPascalCase("my class name"));
        assertEquals("Test", helper.toPascalCase("test"));
    }

    @Test
    void testExtractMethodNames() {
        String code = """
            public class MyClass {
                public void methodOne() {
                }
                
                private int methodTwo(String param) {
                    return 0;
                }
                
                protected static String methodThree() {
                    return null;
                }
            }
            """;
        
        List<String> methods = helper.extractMethodNames(code);
        assertEquals(3, methods.size());
        assertTrue(methods.contains("methodOne"));
        assertTrue(methods.contains("methodTwo"));
        assertTrue(methods.contains("methodThree"));
    }

    @Test
    void testCountLines() {
        assertEquals(3, helper.countLines("line1\nline2\nline3"));
        assertEquals(1, helper.countLines("single line"));
        assertEquals(0, helper.countLines(""));
        assertEquals(0, helper.countLines(null));
    }

    @Test
    void testRemoveSingleLineComments() {
        String code = "int x = 5; // this is a comment\nint y = 10;";
        String result = helper.removeSingleLineComments(code);
        assertEquals("int x = 5; \nint y = 10;", result);
    }

    @Test
    void testIndentCode() {
        String code = "line1\nline2\nline3";
        String result = helper.indentCode(code, 4);
        assertEquals("    line1\n    line2\n    line3", result);
    }

    @Test
    void testGenerateGetterName() {
        assertEquals("getName", helper.generateGetterName("name"));
        assertEquals("getAge", helper.generateGetterName("age"));
        assertEquals("isActive", helper.generateGetterName("isActive"));
        assertEquals("isValid", helper.generateGetterName("isValid"));
        assertEquals("getIs", helper.generateGetterName("is"));
    }

    @Test
    void testGenerateSetterName() {
        assertEquals("setName", helper.generateSetterName("name"));
        assertEquals("setAge", helper.generateSetterName("age"));
        assertEquals("setActive", helper.generateSetterName("isActive"));
        assertEquals("setValid", helper.generateSetterName("isValid"));
        assertEquals("setIs", helper.generateSetterName("is"));
    }
}
