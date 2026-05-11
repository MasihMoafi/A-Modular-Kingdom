flaws: 

### **Non-functional Capabilities**  
1. **File Access**:  
   - Cannot read or process files like `@exc1_1404.pdf` or `@Napoleon.pdf` (content is empty/corrupted).  
   - No ability to access external files unless explicitly provided or uploaded.  

2. **Execution/Computation**:  
   - No execution of code or computational tasks (e.g., `/exec`).  

3. **Memory Management**:  
   - Cannot add new data to memory (`/memory_migrate`, `/memory_cleanup` are placeholders).  
   - Memory is read-only for external knowledge (e.g., `Napoleon.pdf` is corrupted).  

4. **Document Creation**:  
   - No ability to create new files (`/write`, `/read`, `/mkdocx` are not functional).  

5. **System Tools**:  
   - No access to system files or paths (e.g., `/home/masih/...`).  

### **Partial Functionalities**  
- **Browser Automation**: Limited to basic scraping (`/browse_web`) without captcha solving.  
- **Templates**: Can reference formatting templates (`iut-report-template`, `iut-slide-template`) but cannot generate new files.  

## Fix the fucking rendering for math and python and everything else. 

## Build a fully-fledged app - if you like - you can create a UI for it as well - add multi-modality via gemma3 - code work via gpt-oss:20b - etc. 

## I need to be able to toggle tools on and off. 
