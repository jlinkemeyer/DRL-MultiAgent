using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MacDisableBlocks : MonoBehaviour
{
    public GameObject blockRed;
    public GameObject blockBlue;
    public GameObject instruction;
    
    // Start is called before the first frame update
    public void SetActive()
    { 
        blockRed.SetActive(gameObject.active); 
        blockBlue.SetActive(gameObject.active);
        instruction.SetActive(gameObject.active);
    }
    
    public void DeactivateBlue()
    { 
        blockRed.SetActive(true); 
        blockBlue.SetActive(false);
    }
    
    public void DeactivateRed()
    { 
        blockRed.SetActive(false); 
        blockBlue.SetActive(true);
    }
}
