// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com
// Edited by Liam Boone for use with CUDA v5.5

#include <iostream>
#include "scene.h"
#include <cstring>

scene::scene(string filename){
	cout << "Reading scene from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if(fp_in.is_open()){
		while(fp_in.good()){
			string line;
            utilityCore::safeGetline(fp_in,line);
			if(!line.empty()){
				vector<string> tokens = utilityCore::tokenizeString(line);
				if(strcmp(tokens[0].c_str(), "MATERIAL")==0){
				    loadMaterial(tokens[1]);
				    cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "OBJECT")==0){
				    loadObject(tokens[1]);
				    cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "LIGHT")==0){
				    loadLight(tokens[1]);
				    cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "VOLUME")==0){
				    loadVolume(tokens[1]);
				    cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "CAMERA")==0){
				    loadCamera();
				    cout << " " << endl;
				}
			}
		}
	}
}

int scene::loadObject(string objectid){
    int id = atoi(objectid.c_str());
    if(id!=objects.size()){
        cout << "ERROR: OBJECT ID does not match expected number of objects" << endl;
        return -1;
    }else{
        cout << "Loading Object " << id << "..." << endl;
        geom newObject;
		newObject.objectid = id;
        string line;
        
        //load object type 
        utilityCore::safeGetline(fp_in,line);
        if (!line.empty() && fp_in.good()){
            if(strcmp(line.c_str(), "sphere")==0){
                cout << "Creating new sphere..." << endl;
				newObject.type = SPHERE;
            }else if(strcmp(line.c_str(), "cube")==0){
                cout << "Creating new cube..." << endl;
				newObject.type = CUBE;
            }else{
				string objline = line;
                string name;
                string extension;
                istringstream liness(objline);
                getline(liness, name, '.');
                getline(liness, extension, '.');
                if(strcmp(extension.c_str(), "obj")==0){
                    cout << "Creating new mesh..." << endl;
                    cout << "Reading mesh from " << line << "... " << endl;
		    		newObject.type = MESH;
                }else{
                    cout << "ERROR: " << line << " is not a valid object type!" << endl;
                    return -1;
                }
            }
        }
       
	//link material
    utilityCore::safeGetline(fp_in,line);
	if(!line.empty() && fp_in.good()){
	    vector<string> tokens = utilityCore::tokenizeString(line);
	    newObject.materialid = atoi(tokens[1].c_str());
	    cout << "Connecting Object " << objectid << " to Material " << newObject.materialid << "..." << endl;
        }
        
	//load frames
    int frameCount = 0;
    utilityCore::safeGetline(fp_in,line);
	vector<glm::vec3> translations;
	vector<glm::vec3> scales;
	vector<glm::vec3> rotations;
    while (!line.empty() && fp_in.good()){
	    
	    //check frame number
	    vector<string> tokens = utilityCore::tokenizeString(line);
        if(strcmp(tokens[0].c_str(), "frame")!=0 || atoi(tokens[1].c_str())!=frameCount){
            cout << "ERROR: Incorrect frame count!" << endl;
            return -1;
        }
	    
	    //load tranformations
	    for(int i=0; i<3; i++){
            glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
            utilityCore::safeGetline(fp_in,line);
            tokens = utilityCore::tokenizeString(line);
            if(strcmp(tokens[0].c_str(), "TRANS")==0){
                translations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "ROTAT")==0){
                rotations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "SCALE")==0){
                scales.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }
	    }
	    
	    frameCount++;
        utilityCore::safeGetline(fp_in,line);
	}
	
	//move frames into CUDA readable arrays
	newObject.translations = new glm::vec3[frameCount];
	newObject.rotations = new glm::vec3[frameCount];
	newObject.scales = new glm::vec3[frameCount];
	newObject.transforms = new cudaMat4[frameCount];
	newObject.inverseTransforms = new cudaMat4[frameCount];
	for(int i=0; i<frameCount; i++){
		newObject.translations[i] = translations[i];
		newObject.rotations[i] = rotations[i];
		newObject.scales[i] = scales[i];
		glm::mat4 transform = utilityCore::buildTransformationMatrix(translations[i], rotations[i], scales[i]);
		newObject.transforms[i] = utilityCore::glmMat4ToCudaMat4(transform);
		newObject.inverseTransforms[i] = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	}
	
        objects.push_back(newObject);
	
	cout << "Loaded " << frameCount << " frames for Object " << objectid << "!" << endl;
        return 1;
    }
}

int scene::loadVolume(string volumeid){
	
	int id = atoi(volumeid.c_str());
	
	if(id!=volumes.size())
	{
		cout << "ERROR: VOLUME ID does not match expected number of volumes" << endl;
		return -1;
	}
	
	else
	{
		cout << "Loading Volume " << id << "..." << endl;
		volume newVolume;
		newVolume.volumeid = id;
		string line;
        
		// get static data
		for(int i=0; i<4; i++){
			utilityCore::safeGetline(fp_in,line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "mtid")==0){
				newVolume.materialid = atoi(tokens[1].c_str());
				cout << "Connecting Object " << volumeid << " to Material " << newVolume.materialid << "..." << endl;
			}else if(strcmp(tokens[0].c_str(), "delt")==0){
				newVolume.delt = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "step")==0){
				newVolume.step = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "xyzc")==0){
				newVolume.xyzc = glm::vec3(atoi(tokens[1].c_str()), atoi(tokens[2].c_str()), atoi(tokens[3].c_str()));
			}
		}
		
		newVolume.translation = glm::vec3(0.0f);
		newVolume.rotation = glm::vec3(0.0f);
		newVolume.scale = glm::vec3(newVolume.xyzc.x*newVolume.delt, newVolume.xyzc.y*newVolume.delt, newVolume.xyzc.z*newVolume.delt);
		glm::mat4 transform = utilityCore::buildTransformationMatrix(newVolume.translation, newVolume.rotation, newVolume.scale);
		newVolume.transform = utilityCore::glmMat4ToCudaMat4(transform);
		newVolume.inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));

	    //check frame number
	    vector<string> tokens = utilityCore::tokenizeString(line);
		
		// read voxel data
		int numVoxels = newVolume.xyzc.x * newVolume.xyzc.y * newVolume.xyzc.z;
		newVolume.densities = new float[numVoxels];
		for (int i = 0; i < numVoxels; i++) {
			newVolume.densities[i] = 0.0f;
		}

		volumes.push_back(newVolume);
	
		cout << "Loaded Volume " << volumeid << "!" << endl;
			return 1;
	}
}

int scene::loadLight(string lightid){
	int id = atoi(lightid.c_str());
	if(id!=lights.size()){
		cout << "ERROR: LIGHT ID does not match expected number of lights" << endl;
		return -1;
	}else{
		cout << "Loading Light " << id << "..." << endl;
		light newLight;
		newLight.lightid = id;
               
		//load static properties
		for(int i = 0; i < 2; i++){
			string line;
            utilityCore::safeGetline(fp_in,line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "lcol")==0){
				glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newLight.color = color;
			}else if(strcmp(tokens[0].c_str(), "lpos")==0){
				glm::vec3 position( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newLight.position = position;				  
			}
		}
	
		lights.push_back(newLight);
	
		cout << "Loaded Light " << lightid << "!" << endl;
			return 1;
	}
}
int scene::loadCamera(){
	printf("Loading Camera ...\n");
	camera newCamera;
	float fovy;
	
	//load static properties
	for(int i=0; i<8; i++){
		string line;
        utilityCore::safeGetline(fp_in,line);
		vector<string> tokens = utilityCore::tokenizeString(line);
		if(strcmp(tokens[0].c_str(), "brgb")==0){
			newCamera.brgb = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}else if(strcmp(tokens[0].c_str(), "file")==0){
			newCamera.imageName = tokens[1];
		}else if(strcmp(tokens[0].c_str(), "reso")==0){
			newCamera.resolution = glm::vec2(atoi(tokens[1].c_str()), atoi(tokens[2].c_str()));
		}else if(strcmp(tokens[0].c_str(), "iter")==0){
			newCamera.iterations = atoi(tokens[1].c_str());
		}else if(strcmp(tokens[0].c_str(), "eyep")==0){
			newCamera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}else if(strcmp(tokens[0].c_str(), "vdir")==0){
			newCamera.view = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}else if(strcmp(tokens[0].c_str(), "uvec")==0){
			newCamera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}else if(strcmp(tokens[0].c_str(), "fovy")==0){
			fovy = atof(tokens[1].c_str());
		}
	}
    
	//calculate fov based on resolution
	float yscaled = tan(fovy*(PI/180));
	float xscaled = (yscaled * newCamera.resolution.x)/newCamera.resolution.y;
	float fovx = (atan(xscaled)*180)/PI;
	newCamera.fov = glm::vec2(fovx, fovy);

	renderCam = newCamera;
	
	//set up render camera stuff
	renderCam.image = new glm::vec3[(int)renderCam.resolution.x*(int)renderCam.resolution.y];
	for(int i=0; i<renderCam.resolution.x*renderCam.resolution.y; i++){
		renderCam.image[i] = glm::vec3(0,0,0);
	}
	
	cout << "Loaded camera!" << endl;
	return 1;
}

int scene::loadMaterial(string materialid){
	int id = atoi(materialid.c_str());
	if(id!=materials.size()){
		cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
		return -1;
	}else{
		cout << "Loading Material " << id << "..." << endl;
		material newMaterial;
	
		//load static properties
		for(int i=0; i<1; i++){
			string line;
            utilityCore::safeGetline(fp_in,line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "mrgb")==0){
				glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.color = color;
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}
