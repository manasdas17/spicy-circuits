Spicy Circuits
===============

Mobile app that converts a picture of an electronic circuit to a digital SPICE model. Allows users to analyze circuits using digital tools such as Multisim without having to redraw a circuit digitally.

<b>Project Goals:</b>
 * Save time by eliminating the repetion of redrawing a circuit twice
 * Design a system that can be expanded to complete circuit analysis from a phone
 * Connect to other apps that already do mobile circuit analysis
 * Designing a system that allows for new devices to be created and recognized
 
<b>Project index:</b>
 * <b>/spicy-circuits</b> -> Logic that recognizes circuit elements and builds SPICE model (Python)		
 * <b>/app</b> -> Code for mobile app (Currently Android)
 * <b>/pictures</b> -> Pictures used in examples and also used to test our software
 * <b>/docs</b> -> Documentation for Spicy Circuits can be found here
 * <b>/examples</b> -> Examples for accessing Spicy Circuits API and creating new elements

<b>Project Status:</b>
	Spicy Circuits is currently under development. Current goals are to:
 * Get basic recognition of resistors and voltage sources only
 * Formulating an algorithm from preprocessing step to SPICE model generation
 * Get Amazon Virtual Machine working to run logic rather than a computer
 * Get Android app to communicate to Amazon Virtual Machine
 * Find better ways to preprocess image for increased efficiency in object detection